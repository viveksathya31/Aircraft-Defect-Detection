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

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "runs/classify/train4/weights/best.pt"

TEST_DIR = "Data/test"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# ✅ USE YOLO'S CLASS ORDER
class_names = model.names  # dict {id: name}
idx_to_class = {k: v for k, v in class_names.items()}
class_to_idx = {v: k for k, v in class_names.items()}

y_true = []
y_pred = []

# -----------------------------
# RUN INFERENCE
# -----------------------------
for cls_name in os.listdir(TEST_DIR):
    cls_dir = os.path.join(TEST_DIR, cls_name)
    true_idx = class_to_idx[cls_name]

    for img in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, img)

        result = model.predict(img_path, verbose=False)[0]
        pred_idx = int(result.probs.top1)

        y_true.append(true_idx)
        y_pred.append(pred_idx)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# METRICS
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n📊 OVERALL METRICS")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
ordered_names = [idx_to_class[i] for i in range(len(idx_to_class))]

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

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ordered_names,
    yticklabels=ordered_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------
# PER-CLASS ACCURACY
# -----------------------------
print("\n📌 PER-CLASS ACCURACY")
for i, cls in enumerate(ordered_names):
    acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"{cls}: {acc:.4f}")

# -----------------------------
# MISCLASSIFICATION PATTERNS
# -----------------------------
print("\n❌ MISCLASSIFICATION PATTERNS")
for i in range(len(ordered_names)):
    for j in range(len(ordered_names)):
        if i != j and cm[i, j] > 0:
            print(f"{ordered_names[i]} → {ordered_names[j]} : {cm[i, j]}")

from collections import Counter
print(Counter(y_pred))
