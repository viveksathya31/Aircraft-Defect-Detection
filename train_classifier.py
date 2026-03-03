from ultralytics import YOLO

# Load pretrained classification backbone
model = YOLO("yolov8m-cls.pt")  # Medium model (good balance)

# Train on converted dataset
results = model.train(
    data="Yolo_Classify_Data_v2",   # <-- your converted dataset
    epochs=50,
    imgsz=224,
    batch=16,
    device="mps",                   # use Apple GPU
    project="runs/classify",
    name="classifier_from_detection",
    patience=10
)

print("Training complete.")