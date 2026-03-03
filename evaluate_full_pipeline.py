import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DET_MODEL_PATH = "runs/detect/train9/weights/best.pt"
CLS_MODEL_PATH = "/Users/vivek/Projects/Aircraft_Defect_Detection/runs/classify/runs/classify/classifier_from_detection/weights/best.pt"

TEST_IMAGES_DIR = "Yolo_Detection_Data/test/images"
TEST_LABELS_DIR = "Yolo_Detection_Data/test/labels"

IOU_THRESHOLD = 0.5

# -----------------------------
# LOAD MODELS
# -----------------------------
det_model = YOLO(DET_MODEL_PATH)
cls_model = YOLO(CLS_MODEL_PATH)

class_names = cls_model.names

y_true = []
y_pred = []

# -----------------------------
# IOU FUNCTION
# -----------------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area

# -----------------------------
# EVALUATION LOOP
# -----------------------------
image_files = [f for f in os.listdir(TEST_IMAGES_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"🔄 Evaluating {len(image_files)} images...")

for img_file in tqdm(image_files):

    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    label_path = os.path.join(TEST_LABELS_DIR,
                              os.path.splitext(img_file)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # ---- Read Ground Truth ----
    gt_boxes = []
    gt_classes = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id = int(parts[0])

        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        gt_boxes.append([x1, y1, x2, y2])
        gt_classes.append(cls_id)

    # ---- Run Detection ----
    det_results = det_model.predict(img_path, conf=0.25, verbose=False)
    pred_boxes = det_results[0].boxes.xyxy.cpu().numpy()

    matched_gt = set()

    # ---- Match detection with GT using IoU ----
    for pred_box in pred_boxes:

        best_iou = 0
        best_gt_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            iou = compute_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= IOU_THRESHOLD:
            matched_gt.add(best_gt_idx)

            # Crop detected region
            x1, y1, x2, y2 = map(int, pred_box)
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            cls_results = cls_model.predict(crop, verbose=False)
            pred_cls = int(cls_results[0].probs.top1)

            true_cls = gt_classes[best_gt_idx]

            y_true.append(true_cls)
            y_pred.append(pred_cls)

# -----------------------------
# FINAL METRICS
# -----------------------------
if len(y_true) == 0:
    print("❌ No matched detections found.")
    exit()

accuracy = accuracy_score(y_true, y_pred)

print("\n📊 FULL PIPELINE RESULTS")
print(f"Matched Samples: {len(y_true)}")
print(f"Accuracy: {accuracy:.4f}")

print("\n📋 Classification Report")
print(classification_report(
    y_true,
    y_pred,
    target_names=[class_names[i] for i in range(len(class_names))],
    zero_division=0
))