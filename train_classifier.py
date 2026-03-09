from ultralytics import YOLO


model = YOLO("yolov8m-cls.pt")  


results = model.train(
    data="Yolo_Classify_Data_v2",   
    epochs=50,
    imgsz=224,
    batch=16,
    device="mps",                   
    project="runs/classify",
    name="classifier_from_detection",
    patience=10
)

print("Training complete.")