import os
import cv2
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
MODEL_PATH = "runs/classify/runs/classify/classifier_from_detection/weights/best.pt"
TEST_IMAGES_DIR = "Yolo_Detection_Data/test/images"
TEST_LABELS_DIR = "Yolo_Detection_Data/test/labels"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)
class_names = model.names

y_true = []
y_pred = []

image_files = [
    f for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"🔄 Evaluating classifier on {len(image_files)} detection test images...")

# -----------------------------
# LOOP THROUGH TEST IMAGES
# -----------------------------
for img_file in tqdm(image_files):

    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    label_path = os.path.join(
        TEST_LABELS_DIR,
        os.path.splitext(img_file)[0] + ".txt"
    )

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        true_cls = int(parts[0])

        # YOLO normalized → pixel coords
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        results = model.predict(crop, verbose=False)
        pred_cls = int(results[0].probs.top1)

        y_true.append(true_cls)
        y_pred.append(pred_cls)

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
plt.title("Classifier Performance on Detection Test Set")
plt.tight_layout()
plt.savefig("classifier_on_detection_test_confusion_matrix.png")

print("\n✅ Confusion matrix saved as classifier_on_detection_test_confusion_matrix.png")