from ultralytics import YOLO

model = YOLO("yolov8s-cls.pt")

model.train(
    data="Data",
    epochs=100,
    imgsz=224,
    device="mps",
    batch=8,
    lr0=0.0005,
    dropout=0.25,
    augment=True,
    auto_augment="randaugment",
    erasing=0.6,
    val=False
)
print("Training complete.")
