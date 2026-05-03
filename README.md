# 🚗 DriverSafe — Real-Time Driver Drowsiness Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen)
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
| **Phase 3** | MediaPipe + CNN ensemble for maximum accuracy | 🔄 In Progress |
| **Phase 4** | Flutter Android app with TFLite — Play Store deployment | ⏳ Upcoming |

---

## 📂 Repository Structure

```
DriverSafe/
├── phase1/
│   └── drivesafe_phase1.py       # MediaPipe EAR based detection
├── phase2/
│   └── DriveSafe_Phase2.ipynb    # CNN training notebook (Google Colab)
├── models/
│   └── drivesafe_float16.tflite  # Trained model — TFLite export (513 KB)
├── requirements.txt              # Phase 1 dependencies
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
- Framework: TensorFlow 2.19
- Epochs: 30
- Batch size: 64
- Best epoch: 25

---

## 🛠️ Phase 1 Setup

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
pip install -r requirements.txt
```

### Run

```bash
python phase1/drivesafe_phase1.py
```

Press **Q** to quit the webcam window.

> **Note:** Optionally place an `alarm.wav` file in the `phase1/` folder for a custom alarm sound. If absent, a 440Hz beep is auto-generated.

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| MediaPipe 0.10.14 | Face mesh + landmark detection |
| OpenCV | Webcam capture + frame processing |
| NumPy | EAR math calculations |
| Pygame | Audio alarm |
| TensorFlow 2.19 | CNN model training |
| TFLite | On-device mobile inference |
| Flutter *(Phase 4)* | Android app |

---

## 📊 Achieved vs Target Accuracy

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | > 95% | **99.71%** ✅ |
| AUC | > 0.98 | **0.9999** ✅ |
| Inference speed | Real-time (24+ fps) | ✅ |
| Internet required | None | ✅ |

---

## 🔭 What's Coming Next

Phase 3 combines the **MediaPipe EAR algorithm + trained CNN model** into a single ensemble system for maximum accuracy. Both methods must agree on drowsiness before the alarm fires — reducing false positives significantly. After that, Phase 4 brings everything to Android via Flutter.

---

## 👨‍💻 Author

**Parth Kunkunkar**
🔗 [LinkedIn](https://www.linkedin.com/in/parthkunkunkar/)

---

> *This is not a tutorial project. This is a real system being built for real drivers.*
