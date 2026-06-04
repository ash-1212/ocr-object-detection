# Project 4: Computer Vision — OCR + Object Detection 👁️

## 👤 Intern Details
- **Name:** Ayesha Nazish
- **Organization:** DecodeLabs
- **Batch:** 2026
- **Project:** Project 4 — Optional Mastery Project

---

## 🎯 Project Objective
Graduate from structured data processing to machine
perception — teaching computers to READ text from images
and DETECT objects using pre-trained deep learning models.

---

## 🔗 Connection to Previous Projects

| Project | Data Type | Core Task |
|---|---|---|
| Project 1 | Text strings | Rule matching |
| Project 2 | CSV numbers | Classification |
| Project 3 | Text tags | Similarity scoring |
| **Project 4** | **Raw images** | **Visual perception** |

**The progression:** Text → Numbers → Tags → Raw Pixels

---

## 📁 Project Structure
Task4/
├── ocr_reader.py              ← Path 1: OCR code
├── object_detector.py         ← Path 2: Detection code
├── validation_report.py       ← Validation checker
├── images/
│   ├── test_ocr.png           ← OCR test image
│   └── test_detection.jpg     ← Detection test image
├── outputs/
│   ├── preprocessing_pipeline.png
│   ├── ocr_result.png
│   ├── detection_result.png
│   ├── detection_annotated.jpg
│   └── validation_report.png
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
└── README.md

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install opencv-python pytesseract Pillow numpy matplotlib
```
Also install Tesseract OCR engine from:
https://github.com/UB-Mannheim/tesseract/wiki

### Run Path 1 — OCR
```bash
cd e:\Decodelab_AI\Task4
python ocr_reader.py
```

### Run Path 2 — Object Detection
```bash
python object_detector.py
```

### Run Validation
```bash
python validation_report.py
```

---

## ✅ Validation Results

| Rule | Check | Path 1 OCR | Path 2 Detection |
|---|---|---|---|
| Rule 1 | Library Integration | ✅ PASSED | ✅ PASSED |
| Rule 2 | Pre-Processing | ✅ PASSED | ✅ PASSED |
| Rule 3 | Confidence ≥ 80% | ✅ PASSED | ✅ PASSED |
| Rule 4 | Visual Confirmation | ✅ PASSED | ✅ PASSED |

**Total: 8/8 checks passed**

---

## 🧠 Path 1 — OCR Details

| Property | Value |
|---|---|
| Library | pytesseract |
| Engine | Google Tesseract v5 |
| Pre-processing | Grayscale → Gaussian Blur → Adaptive Threshold |
| PSM Mode | 6 (single text block) |
| Output | Extracted text + confidence scores |

### OCR Pipeline
Raw Image → Grayscale → Gaussian Blur (5x5)
→ Adaptive Threshold → Tesseract OCR → Text + Confidence

---

## 🧠 Path 2 — Object Detection Details

| Property | Value |
|---|---|
| Model | MobileNet-SSD |
| Framework | Caffe (via OpenCV DNN) |
| Training Data | PASCAL VOC (20 classes) |
| Input Size | 300 x 300 pixels |
| Pre-processing | Blob: scale=0.007843, mean=127.5 |

### Detection Pipeline
Raw Image → Resize 300x300 → Blob Creation
→ Forward Pass → NMS → Bounding Boxes + Labels

### Detectable Classes
aeroplane, bicycle, bird, boat, bottle, bus, car, cat,
chair, cow, diningtable, dog, horse, motorbike, person,
pottedplant, sheep, sofa, train, tvmonitor

---

## 🔑 Key Concepts Demonstrated

- **OCR** — Optical Character Recognition pipeline
- **Pre-processing** — Grayscale, Blur, Thresholding
- **Transfer Learning** — Using pre-trained models
- **DNN Module** — OpenCV deep neural network loading
- **Blob creation** — Image pre-processing for networks
- **Confidence scoring** — Model certainty measurement
- **Bounding boxes** — Visual object localization
- **NMS** — Non-maximum suppression for clean results

---

## 💡 What I Learned

- How OCR converts pixels to text mathematically
- Why pre-processing is critical for accuracy
- What pre-trained models are and how to use them
- How MobileNet-SSD detects objects in single pass
- How to use OpenCV's DNN module for deep learning
- How confidence thresholds filter weak predictions
- The difference between structured and visual AI data

---

*Built with ❤️ at DecodeLabs AI Industrial Training — Batch 2026*
