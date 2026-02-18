from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="Yolo_Detection_Data/data.yaml",
    epochs=100,
    imgsz=512,
    batch=4,
    plots = False,
    device="mps",
    workers=0,
    max_det=100
)


print("Training complete.")


