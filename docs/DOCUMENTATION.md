# Project Documentation — HackZen 2026 Open Challenge

## 1. Project Title
**Real-Time Driver/Student Drowsiness Detection System**

## 2. Team Details
| Name | Role | GitHub / Contact |
|---|---|---|
| *Sree V K* | Team Lead / CV Developer | *@Sree-26* |
| *Swathi M* | Developer | *@swathim0406* |

## 3. Problem Statement
Drowsy driving is a leading cause of road accidents worldwide, and fatigue
during late-night studying reduces focus and retention. Most people are
unaware they are becoming drowsy until reaction time has already degraded.
There is a need for a low-cost, real-time, camera-based system that can
detect early physiological signs of drowsiness — such as prolonged eye
closure and yawning — and alert the user immediately, using only a standard
webcam with no specialized hardware.

## 4. Objective
To design and implement a real-time Computer Vision system that:
1. Captures live video from a webcam.
2. Detects facial landmarks accurately and efficiently.
3. Computes quantitative drowsiness indicators (EAR, MAR).
4. Triggers a timely alert (visual + audio) when drowsiness is detected.
5. Logs session data for review and future analysis.

## 5. Proposed Solution
We propose a lightweight pipeline built on **MediaPipe Face Mesh**, which
provides 468 3D facial landmarks per frame without requiring any custom
model training. From these landmarks we derive two key metrics:

- **Eye Aspect Ratio (EAR)** — quantifies how open or closed the eyes are.
  A sustained drop below a calibrated threshold indicates the eyes have
  been closed for an unsafe duration.
- **Mouth Aspect Ratio (MAR)** — quantifies mouth opening, used to detect
  yawning as a secondary fatigue indicator.

A short **calibration phase** at the start of each session measures the
user's own baseline (eyes-open) EAR, and sets a personalized threshold at
75% of that baseline. This makes the system robust across different face
shapes, camera angles, and lighting conditions, rather than relying on a
single fixed threshold for everyone.

When drowsiness is detected, the system:
- Displays a red on-screen warning banner
- Plays an audible alert sound
- Logs the event with a timestamp
- Updates a running count of drowsy events and yawns for the session

## 6. Technologies Used
| Category | Tool / Library |
|---|---|
| Programming Language | Python 3.10+ |
| Facial Landmark Detection | MediaPipe Face Mesh (Google, pretrained) |
| Video I/O & Rendering | OpenCV |
| Numerical Computation | NumPy, SciPy |
| Alerts | winsound (Python built-in, Windows only) |
| Optional Web Dashboard | Streamlit |
| Data Logging | Python `csv` module |
| Version Control | Git / GitHub |

> **Note:** Audio alerts use Python's built-in `winsound` module (Windows only). On macOS/Linux, this falls back to a terminal beep (`\a`).

## 7. Dataset
No labeled training dataset is required for the core MVP — the system uses
Google's pretrained MediaPipe Face Mesh model directly via inference,
licensed under Apache 2.0.

For the optional future extension (a supplementary CNN-based eye-state
classifier), the following public datasets are recommended:
- **MRL Eye Dataset** — large-scale open/closed eye image dataset
- **NTHU Drowsy Driver Detection Dataset** — labeled drowsy/non-drowsy driver video dataset

Both are cited here for attribution purposes; neither is required to run the
current MVP.

## 8. Methodology / Model Architecture

### Pipeline Overview
```
Webcam Frame Capture (OpenCV)
        │
        ▼
Face Mesh Inference (MediaPipe) → 468 landmarks
        │
        ├── Eye landmark subset ──► EAR calculation
        │                             │
        │                             ▼
        │                    Consecutive-frame counter
        │                             │
        │                             ▼
        │                    Drowsy alert if EAR < threshold
        │                    for N consecutive frames
        │
        └── Mouth landmark subset ─► MAR calculation
                                      │
                                      ▼
                             Yawn detected if MAR > threshold
        │
        ▼
Overlay (EAR/MAR/stats) + Alert (sound + banner) + CSV Logger
```

### Key Formulas
**Eye Aspect Ratio:**
```
EAR = (‖p2 - p6‖ + ‖p3 - p5‖) / (2 × ‖p1 - p4‖)
```
where p1–p6 are six 2D landmark points around a single eye.

**Mouth Aspect Ratio:**
```
MAR = vertical mouth opening distance / horizontal mouth width
```

### Thresholds (defaults, user-calibratable)
| Parameter | Default | Description |
|---|---|---|
| EAR threshold | 0.25 (or 75% of calibrated baseline) | Below this = eyes considered closed |
| Consecutive frames | 20 | Frames eyes must stay closed before alert triggers |
| MAR threshold | 0.60 | Above this = yawn detected |

## 9. Installation & Setup Instructions
```bash
git clone https://github.com/<your-username>/drowsiness-detection.git
cd drowsiness-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Place a short `.wav` alert sound at `assets/alert.wav` before running.

## 10. Usage Instructions
```bash
# Standard run with calibration
python src/drowsiness_detector.py

# Custom thresholds, skip calibration
python src/drowsiness_detector.py --ear-threshold 0.22 --mar-threshold 0.65 --no-calibration

# Streamlit dashboard version
streamlit run src/app_streamlit.py
```
Press `q` to exit the OpenCV window. Session logs are written automatically
to `logs/session_summary.csv` and `logs/events_<timestamp>.csv`.

## 11. Results and Outputs
- Achieved real-time performance (~20–30 FPS) on a standard CPU-only laptop webcam.
- Drowsiness correctly flagged within ~1 second of sustained eye closure in manual testing.
- Yawn detection reliably triggered on deliberate wide mouth-opening test cases.
- Per-session CSV logs capture individual timestamped events plus an aggregate summary (session duration, drowsy event count, yawn count).

## 12. Future Scope
- Head-pose estimation for detecting nodding/head-drop as an additional signal
- On-device TFLite model for a mobile/embedded (in-vehicle) version
- Integration with vehicle alert systems (dashboard lights, seat vibration motors)
- Multi-face tracking for classroom-scale fatigue monitoring
- Fleet-management web dashboard aggregating driver fatigue data over time
- Training a supplementary CNN eye-state classifier on the MRL Eye Dataset for improved robustness in poor lighting or with eyeglasses/sunglasses

## 13. References
1. Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop.
2. Google MediaPipe Face Mesh — https://github.com/google/mediapipe
3. OpenCV Documentation — https://docs.opencv.org/
4. MRL Eye Dataset (reference for future extension)
5. NTHU Drowsy Driver Detection Dataset (reference for future extension)
