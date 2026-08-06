# 🛡️ SurakshaDrive — Driver Drowsiness Detection

![Flutter](https://img.shields.io/badge/Flutter-3.29.3-blue?logo=flutter)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Android](https://img.shields.io/badge/Android-14-green?logo=android)
![Status](https://img.shields.io/badge/Status-Phase%204B%20Complete-brightgreen)

> **Suraksha (सुरक्षा) = Safety in Hindi.**
> A real-time driver drowsiness detection app built for Indian gig economy drivers — Uber, Ola, Rapido — who drive long shifts with zero safety net. Runs **100% offline**. No internet required.

---

## 🚨 The Problem

Thousands of road accidents every year are caused by driver fatigue. Gig economy drivers often drive 10–12 hour shifts with no safety mechanism in place. SurakshaDrive aims to fix that with a lightweight, offline-first AI system that monitors driver alertness in real time.

> *Previously known as DriveSafe — renamed to SurakshaDrive to better connect with Indian drivers.*

---

## 🗺️ Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | MediaPipe + EAR algorithm — laptop webcam prototype | ✅ Complete |
| **Phase 2** | Custom CNN training on MRL Eye Dataset (48,000 images) | ✅ Complete |
| **Phase 3** | MediaPipe + CNN ensemble — dual verification system | ✅ Complete |
| **Phase 4A** | Flutter Android app — complete UI with all screens | ✅ Complete |
| **Phase 4B** | ML Kit EAR + CNN integration, alert wiring, analytics, settings | ✅ Complete |
| **Phase 4C** | Google Maps integration + background service | ⏳ Upcoming |
| **Phase 4D** | Play Store deployment | ⏳ Upcoming |

---

## 📂 Repository Structure

SurakshaDrive/
├── ml/
│ └── phases/
│ ├── phase1/
│ │ ├── drivesafe_phase1.py # MediaPipe EAR detection
│ │ └── drivesafe_phase3.py # EAR + CNN ensemble
│ ├── phase2/
│ │ └── DriveSafe_Phase2.ipynb # CNN training (Google Colab)
│ ├── models/
│ │ └── drivesafe_float16.tflite # Trained model — 513 KB
│ ├── requirements.txt
│ └── README.md
├── lib/ # Flutter app source
│ ├── main.dart
│ ├── theme.dart # AppColors — light/dark
│ ├── screens/
│ │ ├── splash_screen.dart
│ │ ├── onboarding_screen.dart
│ │ ├── main_screen.dart # Bottom navigation
│ │ ├── home_screen.dart # Camera + EAR + CNN feed
│ │ ├── alert_screen.dart # जागो! रुको! alert
│ │ ├── analytics_screen.dart # Session history + insights
│ │ └── settings_screen.dart # EAR sensitivity, sound, vibration
│ └── services/
│ ├── detection_service.dart # ML Kit face detection + EAR
│ ├── cnn_service.dart # TFLite CNN in background isolate
│ └── analytics_service.dart # Session tracking + SharedPreferences
├── assets/
│ ├── models/ # TFLite model
│ ├── audio/ # Alarm sound
│ └── icon/ # App icon
├── pubspec.yaml
└── README.md

---

## 📱 Phase 4B — Live Detection on Android

### What's Working

| Feature | Status |
|---------|--------|
| Real-time face detection (ML Kit) | ✅ |
| EAR calculation from eye contours | ✅ |
| CNN model inference (background isolate) | ✅ |
| Auto-navigation to Alert screen on drowsiness | ✅ |
| Alarm sound on detection | ✅ |
| Vibration pattern on detection | ✅ |
| EAR sensitivity slider (adjustable threshold) | ✅ |
| Sound / vibration toggle in settings | ✅ |
| Session analytics (avg EAR, alerts, sessions) | ✅ |
| 100% offline operation | ✅ |
| ~24–30 FPS on OnePlus Nord CE 2 Lite 5G | ✅ |

### Detection Architecture

Camera Frame (NV21)
↓
Copy Y+U+V bytes immediately
↙ ↘
ML Kit Face Detection CNN Isolate (background)
EAR from eye contours TFLite inference on eye crop
↓ ↓
EAR < threshold? Score displayed
36→20 frames? (Experimental)
↓
Alert Screen Auto-Navigation
Alarm + Vibration
↓
Session saved to Analytics

### CNN Model Notes
- Trained on MRL Eye Dataset (lab-controlled images, 99.71% accuracy)
- Real-world accuracy varies with lighting and face orientation
- Marked as **Experimental** in the UI — EAR drives detection, CNN is secondary signal
- Runs in a persistent background `Isolate` to avoid blocking the camera stream

---

## 📱 Phase 4A — Flutter App Screens

| Screen | Description |
|--------|-------------|
| **Splash** | Animated eye logo with warm glow |
| **Onboarding** | 4 slides — AI detection, privacy, alerts, battery |
| **Home** | Live camera feed, EAR value, CNN score (collapsible) |
| **Alert** | Full red screen — जागो! रुको! + vibration + alarm |
| **Analytics** | Session history, avg EAR, total alerts, session count |
| **Settings** | Dark mode, EAR sensitivity slider, sound/vibration toggles, language |

### Design System

| Property | Value |
|----------|-------|
| Primary color | Saffron `#FF9500` |
| Light background | `#F2F2F7` |
| Dark background | `#1C1C1E` slate |
| Card surface | `#FFFFFF` / `#2C2C2E` |
| Safe color | `#30D158` green |
| Alert color | `#FF453A` red |
| Font | Inter (Google Fonts) |

---

## 🧠 Phase 2 — CNN Model Results

| Metric | Result |
|--------|--------|
| Dataset | MRL Eye Dataset — 48,000 images |
| Test Accuracy | **99.71%** |
| Test AUC | **0.9999** |
| Model size (TFLite float16) | **513 KB** |
| Training platform | Google Colab T4 GPU |
| Best epoch | 25 / 30 |

---

## 🔗 Phase 3 — EAR + CNN Ensemble (Python)
Webcam Frame
↓
MediaPipe Face Mesh (468 landmarks)
↓
Extract Eye Region
↙ ↘
EAR CNN Model (TFLite)
Algorithm 513KB on-device
↓ ↓
EAR < 0.20? CNN < threshold?
↘ ↙
Either triggers?
↓
Alarm + Warning


**Running at 30 FPS on laptop CPU. Zero false alarms after threshold tuning.**

---

## 🛠️ Setup

### Phase 1 — Laptop Webcam (VS Code)

```bash
git clone https://github.com/parthrkunkunkar-ds/SurakshaDrive.git
cd SurakshaDrive

py -3.10 -m venv .venv
.venv\Scripts\activate

pip install opencv-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4 pygame==2.6.1 tensorflow-cpu==2.15.0 protobuf==4.25.9

python ml/phases/phase1/drivesafe_phase1.py   # Phase 1 — EAR only
python ml/phases/phase1/drivesafe_phase3.py   # Phase 3 — EAR + CNN
```

### Phase 4 — Flutter Android App

```bash
git clone https://github.com/parthrkunkunkar-ds/SurakshaDrive.git
cd SurakshaDrive

flutter pub get
flutter run
```

**Prerequisites:**
- Flutter 3.29.3+
- Android phone with Developer Mode + USB Debugging enabled
- Android SDK 36 (`compileSdk = 36` in `build.gradle.kts`)

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Phase 1 & 3 — laptop detection |
| MediaPipe 0.10.14 | Face mesh + landmark detection |
| OpenCV 4.10 | Webcam capture + frame processing |
| TensorFlow CPU 2.15 | TFLite inference |
| Flutter 3.29.3 | Android app framework |
| Google ML Kit | Face detection + eye contours on Android |
| tflite_flutter 0.12.0 | On-device CNN inference |
| audioplayers | Alarm sound |
| vibration | Haptic feedback |
| shared_preferences | Settings + analytics persistence |
| Google Fonts (Inter) | Typography |
| Google Colab T4 GPU | CNN model training |

---

## 📊 Achieved vs Target

| Metric | Target | Achieved |
|--------|--------|----------|
| CNN Accuracy | > 95% | **99.71%** ✅ |
| AUC | > 0.98 | **0.9999** ✅ |
| Inference speed | 24+ fps | **24–30 FPS** ✅ |
| Model size | < 1MB | **513 KB** ✅ |
| Internet required | None | **None** ✅ |
| Offline operation | Full | **Full** ✅ |

---

## 🔭 What's Coming Next

**Phase 4C** — Play Store deployment.

---

## 👨‍💻 Author

**Parth R. Kunkunkar**
🔗 [LinkedIn](https://www.linkedin.com/in/parthkunkunkar/)
⭐ [GitHub](https://github.com/parthrkunkunkar-ds/SurakshaDrive)

---

> *This is not a tutorial project. This is a real system being built for real drivers.*
>
> *Apni suraksha, apne haath — अपनी सुरक्षा, अपने हाथ*
