# 🚗 DriverSafe — Real-Time Driver Drowsiness Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-red)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen)
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
| **Phase 2** | Custom CNN training on MRL Eye Dataset (48,000 images) via Google Colab | 🔄 In Progress |
| **Phase 3** | MediaPipe + CNN ensemble for maximum accuracy | ⏳ Upcoming |
| **Phase 4** | Flutter Android app with TFLite — Play Store deployment | ⏳ Upcoming |

---

## 📂 Repository Structure

```
DriverSafe/
├── phase1/
│   └── drivesafe_phase1.py      # MediaPipe EAR based detection
├── requirements.txt              # Phase 1 dependencies
└── README.md
```

---

## ⚙️ Phase 1 — How It Works

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
| TensorFlow *(Phase 2+)* | CNN model training |
| TFLite *(Phase 4)* | On-device mobile inference |
| Flutter *(Phase 4)* | Android app |

---

## 📊 Target Accuracy

| Metric | Target |
|--------|--------|
| Accuracy | > 95% |
| AUC | > 0.98 |
| Inference speed | Real-time (24+ fps) |
| Internet required | ❌ None |

---

## 🔭 What's Coming Next

Phase 2 involves training a custom CNN on the **MRL Eye Dataset (48,000 labeled images)** on Google Colab with T4 GPU, then converting to TFLite for mobile deployment. More updates coming soon.

---

## 👨‍💻 Author

**Parth Kunkunkar**
[Linkedin](https://www.linkedin.com/in/parthkunkunkar/)

---

> *This is not a tutorial project. This is a real system being built for real drivers.*
