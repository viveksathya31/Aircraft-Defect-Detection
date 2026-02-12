import cv2
from ultralytics import YOLO

# 1. LOAD BOTH MODELS
detector = YOLO("runs/detect/train9/weights/best.pt")  # Your current detection model
classifier = YOLO("runs/classify/defect_classifier/weights/best.pt")  # Your NEW classification model

# 2. LOAD IMAGE
image_path = "/Users/vivek/Projects/Aircraft_Defect_Detection/aircrafts blades/test/images/40_7_JPG_jpg.rf.5e4449bd214b0b872aad2f4b7dc34f78.jpg"
img = cv2.imread(image_path)
original_img = img.copy()

# 3. RUN DETECTION (Stage 1)
results = detector(img)

for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 4. CROP THE OBJECT (The "ROI")
        # Add a small margin/padding if possible to capture context
        h, w, _ = img.shape
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        
        if crop.size == 0: continue

        # 5. RUN CLASSIFICATION (Stage 2)
        # The classifier looks ONLY at the defect, not the whole wing
        cls_results = classifier(crop, verbose=False)
        
        # Get the top predicted class and confidence
        top_class_id = cls_results[0].probs.top1
        top_class_name = cls_results[0].names[top_class_id]
        top_conf = cls_results[0].probs.top1conf.item()

        # 6. DRAW THE FINAL RESULT
        # Use the BOX from Stage 1, but the LABEL from Stage 2
        label = f"{top_class_name} {top_conf:.2f}"
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show or Save
cv2.imshow("Two-Stage Detection", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()