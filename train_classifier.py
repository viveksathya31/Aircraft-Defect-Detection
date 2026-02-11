from ultralytics import YOLO

# 1. Load a pre-trained classification model
# 'yolov8m-cls.pt' is the Medium version (Good balance of speed/accuracy)
model = YOLO('yolov8m-cls.pt') 

# 2. Train the model
results = model.train(
    data='yolo_classify_data',  # Point to the folder containing train/val/test
    epochs=50,                  # 50 epochs is usually enough for classification
    imgsz=224,                  # Standard classification size (or use 256)
    batch=16,                   # Adjust based on your M4 memory
    device='mps',               # Use Apple Silicon GPU
    project='runs/classify',    # Where to save results
    name='defect_classifier',   # Name of this run
    patience=10                 # Stop early if no improvement
)



print("Training complete.")