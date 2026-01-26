import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load("unet_crack.pth", map_location=device))
model.eval()

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

img_path = "unet_data/images/train/crack/1_11_JPG_jpg.rf.c85f156428d9bad0cbbe177d7469a00e.jpg"
mask_path = "unet_data/masks/train/crack/1_11_JPG_jpg.rf.c85f156428d9bad0cbbe177d7469a00e.png"

img = Image.open(img_path).convert("RGB")
gt_mask = Image.open(mask_path).convert("L")

x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(gt_mask, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Predicted Mask")
plt.imshow(pred > 0.3, cmap="gray")
plt.axis("off")

plt.show()
