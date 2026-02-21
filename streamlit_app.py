
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- CONFIGURATION ---
st.set_page_config(page_title="Aircraft Defect Analyzer", layout="wide")
st.title("✈️ Aircraft Defect Detection & Classification")

# Cache models so they don't reload on every button click
@st.cache_resource
def load_models():
    detector = YOLO("runs/detect/train9/weights/best.pt")
    classifier = YOLO("runs/classify/defect_classifier/weights/best.pt")
    return detector, classifier

detector, classifier = load_models()

# --- SIDEBAR / UPLOAD ---
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an aircraft image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # OpenCV uses BGR, but Streamlit/PIL use RGB. 
    # We'll keep 'img' as BGR for YOLO/CV2 and create an RGB version for display.
    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_img = display_img.copy()

    if st.sidebar.button("Run Analysis"):
        with st.spinner('Detecting and Classifying...'):
            # 1. RUN DETECTION
            results = detector(img)
            detections_found = False

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detections_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 2. CROP
                    h, w, _ = img.shape
                    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    
                    if crop.size == 0: continue

                    # 3. RUN CLASSIFICATION
                    cls_results = classifier(crop, verbose=False)
                    top_class_name = cls_results[0].names[cls_results[0].probs.top1]
                    top_conf = cls_results[0].probs.top1conf.item()

                    # 4. DRAW (On the RGB display image)
                    label = f"{top_class_name} {top_conf:.2f}"
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(output_img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(display_img, use_container_width=True)
            
        with col2:
            st.subheader("Analysis Results")
            if detections_found:
                st.image(output_img, use_container_width=True)
            else:
                st.warning("No defects detected.")
else:
    st.info("Please upload an image in the sidebar to begin.")