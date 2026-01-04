import os

base = "Data/train_balanced"


for cls in os.listdir(base):
    cls_path = os.path.join(base, cls)

    if not os.path.isdir(cls_path):
        continue  # skip .DS_Store and files

    count = len([
        f for f in os.listdir(cls_path)
        if os.path.isfile(os.path.join(cls_path, f))
    ])

    print(f"{cls}: {count}")
