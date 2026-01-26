# import os
# import shutil
# import random

# # =====================
# # CONFIG
# # =====================
# BASE_DIR = "unet_data"
# CLASSES = ["Crack", "Dent"]
# TARGET_DIR = os.path.join(BASE_DIR, "images")

# SPLIT = {
#     "train": 0.7,
#     "val": 0.15,
#     "test": 0.15
# }

# random.seed(42)

# # =====================
# # CREATE TARGET FOLDERS
# # =====================
# for split in SPLIT:
#     for cls in CLASSES:
#         os.makedirs(os.path.join(TARGET_DIR, split, cls.lower()), exist_ok=True)

# # =====================
# # SPLIT EACH CLASS
# # =====================
# for cls in CLASSES:
#     cls_path = os.path.join(BASE_DIR, cls)
#     images = [f for f in os.listdir(cls_path)
#               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

#     random.shuffle(images)
#     total = len(images)

#     train_end = int(SPLIT["train"] * total)
#     val_end = train_end + int(SPLIT["val"] * total)

#     splits = {
#         "train": images[:train_end],
#         "val": images[train_end:val_end],
#         "test": images[val_end:]
#     }

#     for split, files in splits.items():
#         for f in files:
#             src = os.path.join(cls_path, f)
#             dst = os.path.join(TARGET_DIR, split, cls.lower(), f)
#             shutil.copy(src, dst)

#     print(f"{cls}: {total} images → "
#           f"{len(splits['train'])} train, "
#           f"{len(splits['val'])} val, "
#           f"{len(splits['test'])} test")

# print("✅ Image split completed.")




# import os

# BASE_DIR = "unet_data"
# SPLITS = ["train", "val", "test"]
# CLASSES = ["crack", "dent"]

# for split in SPLITS:
#     for cls in CLASSES:
#         path = os.path.join(BASE_DIR, "masks", split, cls)
#         os.makedirs(path, exist_ok=True)

# print("✅ Mask folder structure created successfully.")




# # json to png conversion script
# import os
# import json
# import numpy as np
# from PIL import Image, ImageDraw

# IMAGE_DIR = "unet_data/images/train/crack"
# MASK_DIR = "unet_data/masks/train/crack"

# os.makedirs(MASK_DIR, exist_ok=True)

# for file in os.listdir(IMAGE_DIR):
#     if not file.endswith(".json"):
#         continue

#     json_path = os.path.join(IMAGE_DIR, file)

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     image_path = os.path.join(IMAGE_DIR, data["imagePath"])
#     image = Image.open(image_path)
#     width, height = image.size

#     mask = Image.new("L", (width, height), 0)
#     draw = ImageDraw.Draw(mask)

#     for shape in data["shapes"]:
#         if shape["label"] == "defect":
#             points = [tuple(p) for p in shape["points"]]
#             draw.polygon(points, outline=255, fill=255)

#     mask_name = os.path.splitext(file)[0] + ".png"
#     mask.save(os.path.join(MASK_DIR, mask_name))

# print("✅ Masks generated successfully.")





import os

IMG_DIR = "unet_data/images/train/crack"
MASK_DIR = "unet_data/masks/train/crack"

json_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".json")]

print(f"Found {len(json_files)} annotated images\n")

for json_file in json_files:
    base = json_file.replace(".json", "")
    mask_path = os.path.join(MASK_DIR, base + ".png")

    if not os.path.exists(mask_path):
        print("❌ Missing mask for:", base)
    else:
        print("✅ Mask exists for:", base)
