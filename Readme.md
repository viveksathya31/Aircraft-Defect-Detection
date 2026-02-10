#  Aircraft Defect Detection Using Deep Learning

### YOLOv8 Classification + U-Net Segmentation + Hybrid Model

---

##  Project Overview

Aircraft surface defect detection is a critical task in aviation safety and maintenance. Manual inspection is time-consuming, error-prone, and highly dependent on human expertise. This project proposes an **automated deep learning–based approach** to detect and classify surface defects on aircraft components.

The system is developed in **three stages**:

1. **YOLOv8-based defect classification**
2. **U-Net-based crack segmentation**
3. **Hybrid model combining U-Net and YOLOv8**

The hybrid approach improves **localization, interpretability, and robustness** by first segmenting the defect region and then classifying it.

---

##  Objectives

1. Train a YOLOv8 model to classify aircraft surface defects
2. Train a U-Net model for supervised crack segmentation
3. Integrate both models into a **hybrid inference pipeline**
4. Evaluate performance using standard classification metrics

---

## Defect Classes

* Crack
* Dent
* Corrosion
* Scratch
* Paint Off
* Missing Head

---

##  Project Structure

```
Aircraft_Defect_Detection/
│
├── Data/                     # Classification dataset
│   ├── Train/
│   ├── Valid/
│   └── Test/
│
├── unet_data/                # U-Net training data
│   ├── images/
│   └── masks/
│
├── runs/                     # YOLOv8 training outputs
│
├── model.py                  # U-Net architecture definition
├── train_unet.py             # U-Net training script
├── unet_crack.pth            # Trained U-Net weights
├── unet_infer.py             # U-Net inference (segmentation)
│
├── script.py                 # YOLOv8 training script
├── yolov8s-cls.pt            # Pretrained YOLOv8 classification model
│
├── hybrid_infer.py            # Hybrid (U-Net + YOLOv8) inference pipeline
│
├── dataset.py                # Custom PyTorch dataset loader
├── split.py                  # Dataset splitting utility
├── convert_marks.py          # Mask/annotation preprocessing
│
├── evaluate.py               # YOLOv8 evaluation (metrics)
├── evaluate_visual.py        # Visual evaluation
├── confusion_matrix.png      # Saved evaluation plot
├── count_classes.py          # Dataset class distribution
├── visualize.py              # Visualization utilities
│
├── sample.jpg                # Sample test image
└── README.md                 # Project documentation
```

---

##  File-wise Explanation

###  `model.py`

Defines the **U-Net architecture** used for crack segmentation.
This file is shared by both training and inference to ensure architectural consistency.

---

###  `train_unet.py`

Trains the U-Net model using:

* Crack images
* Corresponding binary masks
* Dice loss function

Outputs:

```
unet_crack.pth
```

---

###  `unet_infer.py`

Runs inference using the trained U-Net model.
Produces a **binary segmentation mask** highlighting crack regions.

Used as **Stage-1** in the hybrid pipeline.

---

###  `script.py`

Trains the **YOLOv8 classification model** using aircraft defect images.
Uses transfer learning from `yolov8s-cls.pt`.

Outputs trained models in:

```
runs/classify/trainX/weights/best.pt
```

---

###  `hybrid_infer.py`

Implements the **hybrid inference pipeline**:

1. Segment defect using U-Net
2. Extract defect Region of Interest (ROI)
3. Classify ROI using YOLOv8
4. Visualize results

Final output:

```
hybrid_output.png
```

---

###  `dataset.py`

Custom PyTorch `Dataset` class for loading image–mask pairs for U-Net training.

---

###  `evaluate.py`

Evaluates YOLOv8 classification performance using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

###  `count_classes.py`

Counts number of images per defect class to analyze **class imbalance**.

---

###  Utility Scripts

* `split.py` – dataset splitting
* `convert_marks.py` – annotation/mask conversion
* `visualize.py` – visualization support

---

##  Pipeline Explanation

###  Stage 1: YOLOv8 Classification

* Input: Aircraft surface image
* Output: Defect class label + confidence

---

###  Stage 2: U-Net Segmentation

* Input: Aircraft image
* Output: Binary crack mask (pixel-level localization)

---

###  Stage 3: Hybrid Model

* U-Net segments defect area
* ROI is extracted from mask
* YOLOv8 classifies only the defect region
* Final output includes:

  * Defect mask
  * Bounding box
  * Class label
  * Confidence score

---

##  How to Run the Hybrid Model

Activate virtual environment:

```bash
source .venv/bin/activate
```

Run hybrid inference:

```bash
python3 hybrid_infer.py Data/Test/Crack/<image_name>.jpg
```

Output:

```
hybrid_output.png
```

---

##  Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Per-class performance


