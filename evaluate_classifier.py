import os
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
MODEL_PATH = "runs/classify/defect_classifier/weights/best.pt"
TEST_DIR = "Yolo_Classify_Data/test"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)
class_names = model.names  # {0: 'dent', 1: 'crack', ...}

y_true = []
y_pred = []

# -----------------------------
# COLLECT ALL TEST IMAGES
# -----------------------------
image_paths = []
for class_name in os.listdir(TEST_DIR):
    class_folder = os.path.join(TEST_DIR, class_name)
    if os.path.isdir(class_folder):
        for img in os.listdir(class_folder):
            image_paths.append(os.path.join(class_folder, img))

print(f"🔄 Evaluating on {len(image_paths)} images...")

# -----------------------------
# RUN INFERENCE
# -----------------------------
for img_path in tqdm(image_paths):

    # True label from folder name
    true_class_name = os.path.basename(os.path.dirname(img_path))
    true_class_id = list(class_names.values()).index(true_class_name)
    y_true.append(true_class_id)

    # Prediction
    results = model.predict(img_path, verbose=False)
    pred_class_id = int(results[0].probs.top1)
    y_pred.append(pred_class_id)

# Convert to numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# METRICS
# -----------------------------
ordered_names = [class_names[i] for i in range(len(class_names))]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n📊 OVERALL METRICS")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n📋 CLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    target_names=ordered_names,
    zero_division=0
))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

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
plt.title("Confusion Matrix (Classification Model)")
plt.tight_layout()
plt.savefig("classification_confusion_matrix.png")

print("\n✅ Confusion matrix saved to classification_confusion_matrix.png")