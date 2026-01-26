from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.images = []
        for f in os.listdir(img_dir):
            if not f.endswith(".jpg"):
                continue
            if os.path.exists(os.path.join(mask_dir, f.replace(".jpg", ".png"))):
                self.images.append(f)

        print(f"✅ Loaded {len(self.images)} image-mask pairs")

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image = Image.open(
            os.path.join(self.img_dir, img_name)
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))
        ).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)
        mask = (mask > 0).float()

        return image, mask
