import os
import cv2
import numpy as np

# --------------------------------------------------
# PATH SETUP (ROBUST, HANDLES SPACES)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_TRAIN_DIR = os.path.join(
    BASE_DIR, "aircrafts blades", "train"
)

IMAGES_DIR = os.path.join(YOLO_TRAIN_DIR, "images")
LABELS_DIR = os.path.join(YOLO_TRAIN_DIR, "labels")

OUT_IMG_DIR = os.path.join(BASE_DIR, "unet_data", "images", "train")
OUT_MASK_DIR = os.path.join(BASE_DIR, "unet_data", "masks", "train")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

print("📂 Reading images from:", IMAGES_DIR)

# --------------------------------------------------
# CONVERSION LOOP
# --------------------------------------------------
for img_name in os.listdir(IMAGES_DIR):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGES_DIR, img_name)
    label_path = os.path.join(
        LABELS_DIR, os.path.splitext(img_name)[0] + ".txt"
    )

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # --------------------------------------------------
    # READ YOLO LABELS (BBOX OR POLYGON)
    # --------------------------------------------------
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                class_id = int(values[0])
                coords = values[1:]

                # CASE 1: YOLOv8 segmentation (polygon)
                if len(coords) > 4:
                    points = np.array(coords).reshape(-1, 2)
                    points[:, 0] *= w
                    points[:, 1] *= h
                    points = points.astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)

                # CASE 2: YOLO bounding box
                else:
                    xc, yc, bw, bh = coords

                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

    # --------------------------------------------------
    # SAVE IMAGE & MASK
    # --------------------------------------------------
    cv2.imwrite(os.path.join(OUT_IMG_DIR, img_name), image)
    cv2.imwrite(
        os.path.join(OUT_MASK_DIR, os.path.splitext(img_name)[0] + ".png"),
        mask * 255
    )

print("✅ YOLO annotations successfully converted to U-Net masks")
