"""
hybrid_infer.py
----------------
Hybrid inference pipeline:
U-Net (segmentation) + YOLOv8 (classification)

Pipeline:
1. Segment defect using U-Net
2. Extract defect ROI
3. Classify ROI using YOLOv8
4. Visualize results
"""

import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO

# Import U-Net inference (already fixed and working)
from unet_infer import predict_mask

# --------------------------------------------------
# LOAD YOLOv8 CLASSIFICATION MODEL (FIXED PATH)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs", "classify", "train5", "weights", "best.pt"
)

yolo_model = YOLO(YOLO_MODEL_PATH)

# --------------------------------------------------
# ROI EXTRACTION FUNCTION
# --------------------------------------------------
def extract_roi(image, mask, padding=10):
    """
    Extracts bounding box around defect mask.
    """
    ys, xs = np.where(mask == 1)

    if len(xs) == 0 or len(ys) == 0:
        return None, None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    roi = image[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

# --------------------------------------------------
# YOLOv8 CLASSIFICATION FUNCTION
# --------------------------------------------------
def classify_roi(roi):
    if roi is None or roi.size == 0:
        return None, None

    result = yolo_model.predict(roi, verbose=False)[0]
    cls_id = int(result.probs.top1)
    conf = float(result.probs.top1conf)

    class_name = yolo_model.names[cls_id]
    return class_name, conf

# --------------------------------------------------
# MAIN HYBRID INFERENCE
# --------------------------------------------------
def hybrid_infer(image_path, save_output=True):

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("❌ Could not read image:", image_path)
        return

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1️⃣ Segmentation (U-Net)
    mask = predict_mask(image)

    # 2️⃣ ROI extraction
    roi, bbox = extract_roi(image, mask)

    if roi is None:
        print("⚠️ No defect detected by U-Net")
        return

    # 3️⃣ Classification (YOLOv8)
    defect_class, confidence = classify_roi(roi)

    print("✅ Hybrid Prediction")
    print("Defect Type :", defect_class)
    print("Confidence  :", f"{confidence:.4f}")

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    overlay = image.copy()

    # Mask overlay (red)
    overlay[mask == 1] = [255, 0, 0]

    # Draw bounding box
    x1, y1, x2, y2 = bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"{defect_class} ({confidence:.2f})"
    cv2.putText(
        overlay,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    if save_output:
        cv2.imwrite("hybrid_output.png", overlay_bgr)
        print("🖼️ Output saved as hybrid_output.png")

    cv2.imshow("Hybrid Output", overlay_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hybrid_infer.py <image_path>")
        sys.exit(0)

    img_path = sys.argv[1]
    hybrid_infer(img_path)
