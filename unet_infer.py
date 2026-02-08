"""
unet_infer.py
----------------
Runs inference using a trained U-Net model and returns
a binary defect segmentation mask.

Used as Stage-1 in the Hybrid YOLOv8 + U-Net pipeline.
"""

import torch
import numpy as np
import cv2
import os

# 🔥 IMPORTANT: import the SAME UNet used in training
from model import UNet

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNET_WEIGHTS = os.path.join(BASE_DIR, "unet_crack.pth")

def load_unet_model(weight_path=UNET_WEIGHTS):
    model = UNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

unet_model = load_unet_model()

# --------------------------------------------------
# PREDICT MASK FUNCTION (USED BY HYBRID PIPELINE)
# --------------------------------------------------
def predict_mask(image, threshold=0.5):
    """
    Input:
        image (np.ndarray): RGB image (H, W, 3)
    Output:
        mask (np.ndarray): Binary mask (H, W)
    """

    original_h, original_w = image.shape[:2]

    # Resize to U-Net input size (must match training)
    img_resized = cv2.resize(image, (256, 256))
    img_resized = img_resized / 255.0

    # Convert to tensor
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = unet_model(img_tensor)

    # Sigmoid + threshold
    mask = torch.sigmoid(pred)[0, 0].cpu().numpy()
    mask = (mask > threshold).astype(np.uint8)

    # Resize mask back to original image size
    mask = cv2.resize(mask, (original_w, original_h))

    return mask

# --------------------------------------------------
# TEST BLOCK (OPTIONAL)
# --------------------------------------------------
if __name__ == "__main__":
    img_path = "sample.jpg"  # replace with test image
    image = cv2.imread(img_path)

    if image is None:
        raise FileNotFoundError("❌ sample.jpg not found")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = predict_mask(image)

    cv2.imwrite("predicted_mask.png", mask * 255)
    print("✅ Mask saved as predicted_mask.png")
