# from ultralytics import YOLO

# model = YOLO("yolov8s.pt")

# model.train(
#     data="Yolo_train_data/data.yaml",   # <-- FIX
#     epochs=100,
#     imgsz=640,                          # recommended for detection
#     device="mps",
#     batch=8,
#     lr0=0.0005,
#     augment=True
# )
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="yolo_data/data.yaml",
    epochs=100,
    imgsz=512,
    batch=4,
    plots = False,
    device="mps",
    workers=0,
    max_det=100
)


print("Training complete.")


