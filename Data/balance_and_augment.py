import os
import random
import shutil
from PIL import Image, ImageEnhance

SRC = "Data/train"
DST = "Data/train_balanced"
TARGET = 250

os.makedirs(DST, exist_ok=True)

def augment_image(img):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    return img

for cls in os.listdir(SRC):
    src_cls = os.path.join(SRC, cls)

    if not os.path.isdir(src_cls):
        continue

    dst_cls = os.path.join(DST, cls)
    os.makedirs(dst_cls, exist_ok=True)

    images = [
        f for f in os.listdir(src_cls)
        if os.path.isfile(os.path.join(src_cls, f))
    ]

    # Downsample if too many
    selected = images.copy()
    if len(selected) > TARGET:
        selected = random.sample(selected, TARGET)

    # Copy originals
    for img_name in selected:
        shutil.copy(
            os.path.join(src_cls, img_name),
            os.path.join(dst_cls, img_name)
        )

    # Upsample if too few
    count = len(selected)
    idx = 0
    while count < TARGET:
        img_name = random.choice(selected)
        img_path = os.path.join(src_cls, img_name)

        img = Image.open(img_path)
        img_aug = augment_image(img)

        new_name = f"aug_{idx}_{img_name}"
        img_aug.save(os.path.join(dst_cls, new_name))

        count += 1
        idx += 1

    print(f"{cls}: {count}")
