import os
import glob
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
# Make sure this points to your trained detection model
MODEL_PATH = "runs/detect/train9/weights/best.pt" 

TEST_IMAGES_DIR = "yolo_data/test/images"
TEST_LABELS_DIR = "yolo_data/test/labels"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# Get class names from the model
class_names = model.names  # {0: 'crack', 1: 'dent', ...}

y_true = []
y_pred = []

# List of valid image extensions
valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

# Get list of all image files
image_files = [
    f for f in os.listdir(TEST_IMAGES_DIR) 
    if os.path.splitext(f)[1].lower() in valid_exts
]

print(f"🔄 Evaluating on {len(image_files)} images...")

# -----------------------------
# RUN INFERENCE
# -----------------------------
for img_file in tqdm(image_files):
    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    
    # 1. FIND GROUND TRUTH (from label file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(TEST_LABELS_DIR, label_file)

    if not os.path.exists(label_path):
        # Skip images with no label (background images) if you only want to test defects
        continue
        
    with open(label_path, "r") as f:
        lines = f.readlines()
        if not lines:
            continue
        # Take the first object's class as the "True" class for the whole image
        # (Assuming 1 major defect per crop)
        true_cls_id = int(lines[0].split()[0])
        y_true.append(true_cls_id)

    # 2. GET PREDICTION (from Model)
    # Run inference
    results = model.predict(img_path, verbose=False)
    result = results[0]

    # Check if any object was detected
    if len(result.boxes) > 0:
        # Get the box with the highest confidence
        # result.boxes.cls contains the class IDs of detected objects
        # result.boxes.conf contains the confidence scores
        best_box_idx = result.boxes.conf.argmax() 
        pred_cls_id = int(result.boxes.cls[best_box_idx].item())
        y_pred.append(pred_cls_id)
    else:
        # If nothing detected, we can mark it as a specific "background" class 
        # OR simply skip it. For this confusion matrix, let's mark it as -1 
        # (which we will filter out or handle as "Missed Detection")
        y_pred.append(-1) 

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Filter out missed detections (-1) if you strictly want to compare Class vs Class
# Or map -1 to a "Background" class if you prefer.
# For now, we only calculate metrics on images where SOMETHING was detected vs GT.
valid_mask = y_pred != -1
if np.sum(~valid_mask) > 0:
    print(f"⚠️ Warning: Model missed detections on {np.sum(~valid_mask)} images.")

# -----------------------------
# METRICS & REPORT
# -----------------------------
# Use model.names to get the ordered list of class names
ordered_names = [class_names[i] for i in range(len(class_names))]

# Only evaluating where we made a prediction (or you can map -1 to a 'Background' class)
# Here we filter to keep the lengths matching for scikit-learn
y_true_filtered = y_true[valid_mask]
y_pred_filtered = y_pred[valid_mask]

if len(y_true_filtered) == 0:
    print("❌ No valid predictions made. Check your model or data.")
    exit()

accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
precision = precision_score(y_true_filtered, y_pred_filtered, average="weighted", zero_division=0)
recall = recall_score(y_true_filtered, y_pred_filtered, average="weighted", zero_division=0)
f1 = f1_score(y_true_filtered, y_pred_filtered, average="weighted", zero_division=0)

print("\n📊 OVERALL METRICS (on detected objects)")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n📋 CLASSIFICATION REPORT")
print(classification_report(
    y_true_filtered,
    y_pred_filtered,
    target_names=ordered_names,
    zero_division=0
))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true_filtered, y_pred_filtered)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ordered_names,
    yticklabels=ordered_names
)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix (Detected Objects)")
plt.tight_layout()
plt.savefig("confusion_matrix_fixed.png")
print("\n✅ Confusion matrix saved to confusion_matrix_fixed.png")
# plt.show() # Uncomment if running locally with display