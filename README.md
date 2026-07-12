# 🚗 Real-Time Driver/Student Drowsiness Detection System

A computer vision system that detects signs of drowsiness — prolonged eye
closure and yawning — from a live webcam feed and raises an audio/visual
alert in real time. Built for the **HackZen 2026 Open Challenge**.

![Status](https://img.shields.io/badge/status-hackathon--build-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Problem Statement

Drowsy driving and studying-while-exhausted are major contributors to road
accidents and poor academic performance. Most people don't realize they're
falling asleep until it's too late. This project builds a **lightweight,
real-time, camera-based drowsiness monitor** that can run on a normal laptop
webcam with no special hardware.

## 🎯 Objective

Build a working, real-time system that:
- Detects facial landmarks from a live video feed
- Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
- Flags drowsiness (prolonged eye closure) and yawning
- Raises an alert (sound + on-screen warning) before the user falls asleep
- Logs session stats for later review

## 💡 Proposed Solution

We use **Google's MediaPipe Face Mesh** to extract 468 facial landmarks per
frame with no model training required. From these landmarks we compute:

- **EAR (Eye Aspect Ratio)** — a well-established metric (Soukupová &
  Čech, 2016) that drops sharply when eyes close.
- **MAR (Mouth Aspect Ratio)** — rises sharply during yawning.

If EAR stays below a threshold for a configurable number of consecutive
frames (default: 20 frames, ~1 second at 20fps), the system marks the user
as drowsy and plays an alert sound. A personalized **calibration step** runs
at startup to set the EAR threshold relative to the individual user's normal
eye-open ratio, improving accuracy across different face shapes.

## 🛠️ Technologies Used

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Face/landmark detection | MediaPipe Face Mesh |
| Video capture & rendering | OpenCV |
| Math / distance calculations | NumPy, SciPy |
| Alerts | playsound (audio) |
| Optional dashboard | Streamlit |
| Logging | CSV (built-in `csv` module) |

## 📊 Dataset

No training dataset is required — MediaPipe Face Mesh is a pretrained model
shipped by Google (Apache 2.0 license). All landmark detection happens
on-device in real time. See [References](#-references) for attribution.


## 🧠 Methodology / Architecture

```
Webcam Frame
     │
     ▼
MediaPipe Face Mesh (468 landmarks)
     │
     ├──► Eye landmarks ──► EAR calculation ──► Eye-closed frame counter ──► Drowsy alert
     │
     └──► Mouth landmarks ─► MAR calculation ─► Yawn detection ──► Yawn counter
     │
     ▼
Overlay stats on frame + Alert sound + Session logger (CSV)
```

**EAR formula:**
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

**MAR formula:**
```
MAR = vertical mouth distance / horizontal mouth distance
```

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/drowsiness-detection.git
cd drowsiness-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add an alert sound file
# Place any short .wav file at assets/alert.wav
```

## ▶️ Usage Instructions

**Run the OpenCV desktop app (with calibration):**
```bash
python src/drowsiness_detector.py
```

**Run with custom thresholds / skip calibration:**
```bash
python src/drowsiness_detector.py --ear-threshold 0.22 --mar-threshold 0.65 --no-calibration
```

**Run the Streamlit dashboard version:**
```bash
streamlit run src/app_streamlit.py
```

Press **`q`** to quit the OpenCV window at any time. Session stats are saved
automatically to `logs/session_summary.csv`.

## 📈 Results and Outputs

- Real-time EAR/MAR values displayed on-screen at ~20–30 FPS on a standard laptop webcam (CPU only, no GPU required)
- Drowsiness correctly flagged within ~1 second of sustained eye closure during testing
- Yawn detection triggers reliably on wide mouth-opening gestures
- Per-session CSV logs: individual event timestamps + summary (duration, drowsy event count, yawn count)

## 🔭 Future Scope

- Head-pose based nodding detection for additional fatigue signal
- Mobile app version (on-device TFLite model for offline use in vehicles)
- Integration with vehicle systems (dashboard alert, seat vibration)
- Multi-face support for classroom-wide monitoring
- Cloud dashboard for fleet managers to review driver fatigue trends
- Adaptive thresholds using a lightweight ML classifier trained on the MRL Eye Dataset for higher accuracy in low light / with glasses

## 📚 References

- Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.*
- [MediaPipe Face Mesh — Google](https://github.com/google/mediapipe)
- [OpenCV Documentation](https://docs.opencv.org/)
- MRL Eye Dataset (optional extension reference)
- NTHU Drowsy Driver Detection Dataset (optional extension reference)

## 👥 Team Details

| Name | Role |
|---|---|
| *Sree V K* | *CV Developer* |
| *Swathi M* | *Developer* |

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

Built for **HackZen 2026 Open Challenge** 🏆
