# 🚗 DriverSafe — Real-Time Driver Drowsiness Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Phase%203%20Complete-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Android-lightgrey?logo=android)

> A real-time driver drowsiness detection system built for gig economy drivers — Uber, Ola, Rapido — who drive long shifts with zero safety net. Runs **100% offline**. No internet required.

---

## 🚨 The Problem

Thousands of road accidents every year are caused by driver fatigue. Gig economy drivers often drive 10–12 hour shifts with no safety mechanism in place. DriverSafe aims to fix that with a lightweight, offline-first AI system that monitors driver alertness in real time.

---

## 🗺️ Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | MediaPipe + EAR algorithm — laptop webcam prototype | ✅ Complete |
| **Phase 2** | Custom CNN training on MRL Eye Dataset (48,000 images) via Google Colab | ✅ Complete |
| **Phase 3** | MediaPipe + CNN ensemble — dual verification system | ✅ Complete |
| **Phase 4** | Flutter Android app with TFLite — Play Store deployment | 🔄 In Progress |

---

## 📂 Repository Structure

```
DriverSafe/
├── phase1/
│   ├── drivesafe_phase1.py       # MediaPipe EAR based detection
│   └── drivesafe_phase3.py       # EAR + CNN ensemble system
├── phase2/
│   └── DriveSafe_Phase2.ipynb    # CNN training notebook (Google Colab)
├── models/
│   └── drivesafe_float16.tflite  # Trained model — TFLite export (513 KB)
├── requirements.txt              # Dependencies
└── README.md
```

---

## ⚙️ Phase 1 — MediaPipe EAR Detection

Phase 1 runs entirely on a laptop webcam using **MediaPipe Face Mesh** and the **Eye Aspect Ratio (EAR)** algorithm.

```
Webcam Frame → MediaPipe Face Mesh (468 landmarks) → Extract 6 Eye Points → Compute EAR → Threshold Check → Alarm
```

### Eye Aspect Ratio (EAR)

```
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
```

- EAR ≈ **0.30** → eyes open
- EAR ≈ **0.0** → eyes closed
- If EAR stays below **0.25** for **48 consecutive frames (~2 seconds)** → drowsiness detected

### On Detection:
- 🔔 Loud audio alarm triggers instantly
- 🚨 Full-screen **"DROWSY! PULL OVER!"** warning appears
- System resets automatically once eyes reopen

---

## 🧠 Phase 2 — Custom CNN Model

Phase 2 trains a custom Convolutional Neural Network on the **MRL Eye Dataset** using Google Colab's T4 GPU, then exports to TFLite for mobile deployment.

### Dataset
| Property | Value |
|----------|-------|
| Dataset | MRL Eye Dataset |
| Total images | 48,000 |
| Classes | `open_eye` / `closed_eye` |
| Class balance | Perfectly balanced (24k each) |

### Model Architecture
- 4 Conv2D blocks with BatchNormalization + MaxPooling
- GlobalAveragePooling2D
- Dense(128) + Dropout(0.4)
- Sigmoid output
- Total parameters: **258,881** (~1MB)

### Results

| Metric | Result |
|--------|--------|
| Test Accuracy | **99.71%** |
| Test AUC | **0.9999** |
| Test Loss | **0.0103** |
| Model size (TFLite float16) | **513 KB** |

### Training Setup
- Platform: Google Colab (T4 GPU)
- Framework: TensorFlow 2.15
- Epochs: 30
- Batch size: 64
- Best epoch: 25

---

## 🔗 Phase 3 — EAR + CNN Ensemble

Phase 3 combines both detection systems into a single real-time pipeline running at **30 FPS**.

```
Webcam Frame
      ↓
MediaPipe Face Mesh
      ↓
Extract Eye Region
   ↙        ↘
EAR          CNN Model (TFLite)
Algorithm    513KB on-device
   ↓               ↓
EAR < 0.25?   CNN < threshold?
   ↘        ↙
  Either triggers?
       ↓
  Alarm + Warning
```

### Why Ensemble?
- **EAR alone** — fast but sensitive to lighting and head angle
- **CNN alone** — accurate but can miss partial closures
- **Combined** — EAR catches geometric closure, CNN catches subtle drooping. Together they eliminate false positives significantly.

### Live Overlay
- `EAR: 0.291` — real-time eye aspect ratio
- `CNN: 0.998` — real-time CNN confidence (1.0 = open, 0.0 = closed)
- `Mode: Eyes Open / EAR only / CNN only / BOTH CLOSED`
- Drowsy meter bar — fills up over 2 seconds before alarm fires
- **30 FPS** on standard laptop CPU

---

## 🛠️ Setup & Run

### Prerequisites
- Python 3.10
- Webcam

### Installation

```bash
# Clone the repo
git clone https://github.com/parthrkunkunkar-ds/DriverSafe.git
cd DriverSafe

# Create virtual environment
py -3.10 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install opencv-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4 pygame==2.6.1 tensorflow-cpu==2.15.0 protobuf==4.25.9
```

### Run Phase 1 (EAR only)
```bash
python phase1/drivesafe_phase1.py
```

### Run Phase 3 (EAR + CNN ensemble)
```bash
python phase1/drivesafe_phase3.py
```

Press **Q** to quit.

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| MediaPipe 0.10.14 | Face mesh + landmark detection |
| OpenCV 4.10 | Webcam capture + frame processing |
| NumPy 1.26 | EAR math calculations |
| Pygame | Audio alarm |
| TensorFlow CPU 2.15 | TFLite inference |
| TFLite float16 | On-device model (513 KB) |
| Flutter *(Phase 4)* | Android app |

---

## 📊 Achieved vs Target

| Metric | Target | Achieved |
|--------|--------|----------|
| CNN Accuracy | > 95% | **99.71%** ✅ |
| AUC | > 0.98 | **0.9999** ✅ |
| Inference speed | 24+ fps | **30 FPS** ✅ |
| Internet required | None | **None** ✅ |
| Model size | < 1MB | **513 KB** ✅ |

---

## 🔭 What's Coming Next

Phase 4 brings everything to Android via **Flutter + TFLite**. The same ensemble logic — MediaPipe face detection + CNN eye classification — will run entirely on the phone's front camera, with vibration + loud alarm on drowsiness detection. Target: Play Store deployment.

---

## 👨‍💻 Author

**Parth Kunkunkar**
🔗 [LinkedIn](https://www.linkedin.com/in/parthkunkunkar/)

---

> *This is not a tutorial project. This is a real system being built for real drivers.*
