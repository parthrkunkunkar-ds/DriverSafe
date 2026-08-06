# 🛡️ SurakshaDrive — Real-Time Driver Drowsiness Detection

![Flutter](https://img.shields.io/badge/Flutter-3.29.3-blue?logo=flutter)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Android](https://img.shields.io/badge/Android-14-green?logo=android)
![Offline](https://img.shields.io/badge/100%25%20Offline-No%20Internet%20Needed-success)

> **Suraksha (सुरक्षा) = Safety.**  
> A real-time, on-device AI system that detects driver drowsiness using computer vision — built entirely in Flutter with zero cloud dependencies.

---

## 🎯 Why I Built This

This is my **first production-attempt mobile app**, built to prove I can ship end-to-end AI on the edge. Most drowsiness detection projects stop at a Python notebook. I took it all the way to a working Android app that runs **100% offline** at 24–30 FPS on a mid-range phone.

Built for Indian gig-economy drivers (Uber, Ola, Rapido) who work 10+ hour shifts with no safety net. No internet required. No API costs. No privacy leaks.

---

## 🚀 What's Working

| Feature | Status | Tech |
|---------|--------|------|
| Real-time face detection | ✅ | Google ML Kit |
| Eye Aspect Ratio (EAR) calculation | ✅ | Custom contour math |
| CNN eye-state classifier | ✅ | TFLite float16, 513 KB |
| Dual verification (EAR + CNN) | ✅ | Ensemble logic |
| Audio + vibration alerts | ✅ | `audioplayers` + `vibration` |
| Adjustable EAR sensitivity | ✅ | Real-time threshold slider |
| Session analytics & persistence | ✅ | SharedPreferences |
| Alert event tracking | ✅ | Persistent counter |
| Dark mode support | ✅ | Dynamic theming |
| 24–30 FPS on OnePlus Nord CE 2 Lite | ✅ | Background isolate |

---

## 🏗️ Architecture

```
Camera Frame (NV21, ~30 FPS)
        │
        ▼
┌─────────────────┐     ┌─────────────────────┐
│  Google ML Kit  │────▶│  CNN Isolate        │
│  Face Mesh      │     │  (TFLite float16)   │
│  → EAR Value    │     │  → Open/Closed Score│
└─────────────────┘     └─────────────────────┘
        │                         │
        └──────────┬──────────────┘
                   ▼
        ┌─────────────────────┐
        │  Drowsiness Logic   │
        │  EAR < threshold    │
        │  for 20 frames      │
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Alert Screen       │
        │  Sound + Vibration  │
        │  Auto-navigation    │
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Analytics Service  │
        │  Session + EAR avg  │
        │  Persisted locally  │
        └─────────────────────┘
```

---

## 🧠 The ML Pipeline

### EAR (Eye Aspect Ratio) — Primary Signal
- Derived from 6 eye landmarks per eye via ML Kit face contours
- Real-time geometric calculation — no model inference needed
- Threshold: adjustable 0.15–0.25 (default 0.20)

### CNN — Secondary Verification
- **Dataset:** MRL Eye Dataset (48,000 images)
- **Accuracy:** 99.71% | **AUC:** 0.9999
- **Model size:** 513 KB (TFLite float16)
- Runs in a **persistent background `Isolate`** — zero UI jank
- Marked as *Experimental* in-app because real-world lighting varies

### Why Two Signals?
EAR alone can false-trigger on blinking. The CNN validates actual eye state. Either can trigger the alert, but EAR drives the primary logic.

---

## 📊 Performance

| Metric | Result |
|--------|--------|
| CNN Test Accuracy | **99.71%** |
| AUC | **0.9999** |
| On-device inference | **24–30 FPS** |
| Model size | **513 KB** |
| Internet required | **None** |
| Cold start to detection | **< 2 seconds** |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Mobile** | Flutter 3.29, Dart |
| **CV / Face Detection** | Google ML Kit (on-device) |
| **Deep Learning** | TensorFlow 2.15 → TFLite float16 |
| **Flutter ML** | `tflite_flutter` with background isolate |
| **State / UI** | StatefulWidget, `ValueKey` rebuilds |
| **Persistence** | `shared_preferences` |
| **Media** | `audioplayers`, `vibration` |
| **Training** | Google Colab T4 GPU |

---

## 📱 Screens

| Screen | Purpose |
|--------|---------|
| **Splash** | Animated eye logo with warm glow |
| **Onboarding** | 4-slide intro — AI, privacy, alerts, battery |
| **Home** | Live camera, real-time EAR + CNN score |
| **Alert** | Full-screen red alert — जागो! रुको! |
| **Analytics** | Session history, avg EAR, total alerts, drive count |
| **Settings** | Dark mode, EAR sensitivity, sound/vibration toggles |

---

## 🗺️ Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | MediaPipe + EAR prototype (Python) | ✅ |
| **Phase 2** | Custom CNN training (Colab) | ✅ |
| **Phase 3** | EAR + CNN ensemble (Python) | ✅ |
| **Phase 4A** | Flutter UI + all screens | ✅ |
| **Phase 4B** | ML Kit + CNN integration, alerts, analytics | ✅ |
| **Phase 4C** | Play Store deployment | ⏳ |

> **Note:** Google Maps integration was intentionally scoped out for v1 to keep the app 100% offline and zero-cost. Location-based drive time tracking is planned for a future production version.

---

## 🧪 Run It Locally

```bash
git clone https://github.com/parthrkunkunkar-ds/SurakshaDrive.git
cd SurakshaDrive
flutter pub get
flutter run
```

**Requirements:**
- Flutter 3.29+
- Android SDK 36
- Physical Android device (camera required)

---

## 🎓 What I Learned

- **On-device ML:** Converting a 99.7% accuracy CNN to a 513 KB TFLite model that runs in a Flutter isolate
- **Real-time CV:** Bridging NV21 camera frames to ML Kit and custom contour math
- **State management:** Coordinating camera lifecycle, alert navigation, and analytics persistence without external state libraries
- **Performance:** Maintaining 24–30 FPS while running dual inference pipelines
- **Product thinking:** Building for users with low-end devices and zero internet

---

## 👨‍💻 Author

**Parth R. Kunkunkar**  
🔗 [LinkedIn](https://www.linkedin.com/in/parthkunkunkar/)  
⭐ [GitHub](https://github.com/parthrkunkunkar-ds)

---

> *Apni suraksha, apne haath — अपनी सुरक्षा, अपने हाथ*  
> *Your safety, in your hands.*
