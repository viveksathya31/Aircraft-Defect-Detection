import os, json
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = (512, 512)

json_dir = "unet_data/images/train/crack"
mask_out = "unet_data/masks/train/crack"

os.makedirs(mask_out, exist_ok=True)

for file in os.listdir(json_dir):
    if not file.endswith(".json"):
        continue

    with open(os.path.join(json_dir, file)) as f:
        data = json.load(f)

    mask = Image.new("L", IMG_SIZE, 0)
    draw = ImageDraw.Draw(mask)

    for shape in data["shapes"]:
        points = [(int(x), int(y)) for x, y in shape["points"]]
        draw.polygon(points, fill=255)

    out_name = file.replace(".json", ".png")
    mask.save(os.path.join(mask_out, out_name))

print("✅ Masks converted to PNG")
