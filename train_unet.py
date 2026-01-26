import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import CrackDataset
from model import UNet

# =====================
# Device
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Dice Loss & Metric
# =====================
def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    return 1 - (2 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.3).float()   # 🔥 LOWER THRESHOLD

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    return (2 * intersection + smooth) / (union + smooth)

# =====================
# Datasets
# =====================
train_dataset = CrackDataset(
    "unet_data/images/train/crack",
    "unet_data/masks/train/crack"
)

val_dataset = CrackDataset(
    "unet_data/images/train/crack",
    "unet_data/masks/train/crack"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# =====================
# Model
# =====================
model = UNet().to(device)
criterion = dice_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =====================
# Training
# =====================
epochs = 20

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    dice_total = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            dice_total += dice_score(preds, masks).item()

    avg_dice = dice_total / len(val_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss/len(train_loader):.4f} "
        f"Val Dice: {avg_dice:.4f}"
    )

# =====================
# Save model
# =====================
torch.save(model.state_dict(), "unet_crack.pth")
print("✅ Model saved")
