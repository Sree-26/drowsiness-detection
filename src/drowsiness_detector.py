"""
Real-Time Driver/Student Drowsiness Detection System
------------------------------------------------------
Detects drowsiness using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
computed from MediaPipe Face Mesh landmarks. Triggers an audio + visual alert
when prolonged eye closure or yawning is detected.

Author: Sree V K
"""

import cv2
import time
import argparse
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

from utils import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    LEFT_EYE, RIGHT_EYE, MOUTH,
    play_alert_sound,
)
from logger import SessionLogger


class DrowsinessDetector:
    def __init__(
        self,
        ear_threshold: float = 0.25,
        ear_consec_frames: int = 20,
        mar_threshold: float = 0.6,
        camera_index: int = 0,
    ):
        self.ear_threshold = ear_threshold
        self.ear_consec_frames = ear_consec_frames
        self.mar_threshold = mar_threshold
        self.camera_index = camera_index

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.eye_closed_frames = 0
        self.yawn_count = 0
        self.drowsy_events = 0
        self.alert_active = False
        self.yawn_active = False
        self.logger = SessionLogger()

    def calibrate(self, cap, seconds: int = 5):
        """Measure the user's baseline (eyes-open) EAR to personalize threshold."""
        print(f"[INFO] Calibrating for {seconds} seconds. Please look at the camera with eyes open...")
        start = time.time()
        ear_values = []

        while time.time() - start < seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                ear_values.append((left_ear + right_ear) / 2.0)

            cv2.putText(frame, "Calibrating... Keep eyes open", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Drowsiness Detection - Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyWindow("Drowsiness Detection - Calibration")

        if ear_values:
            baseline = np.mean(ear_values)
            self.ear_threshold = round(baseline * 0.75, 3)  # 75% of open-eye EAR
            print(f"[INFO] Calibration complete. Baseline EAR: {baseline:.3f} | "
                  f"New threshold: {self.ear_threshold}")
        else:
            print("[WARN] Calibration failed, using default threshold.")

    def run(self, use_calibration: bool = True):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not access webcam. Check camera index / permissions.")

        if use_calibration:
            self.calibrate(cap)

        print("[INFO] Starting detection. Press 'q' to quit.")
        session_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            h, w, _ = frame.shape
            status_text = "Awake"
            status_color = (0, 200, 0)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                mar = mouth_aspect_ratio(landmarks, MOUTH, w, h)

                # --- Eye closure logic ---
                if avg_ear < self.ear_threshold:
                    self.eye_closed_frames += 1
                    if self.eye_closed_frames >= self.ear_consec_frames:
                        status_text = "DROWSY - EYES CLOSED!"
                        status_color = (0, 0, 255)
                        if not self.alert_active:
                            self.drowsy_events += 1
                            self.logger.log_event("drowsy_eyes_closed")
                            play_alert_sound()
                            self.alert_active = True
                else:
                    self.eye_closed_frames = 0
                    self.alert_active = False

                # --- Yawn logic ---
                if mar > self.mar_threshold:
                    status_text = "YAWNING DETECTED"
                    status_color = (0, 165, 255)
                    if not self.yawn_active:
                        self.yawn_count += 1
                        self.logger.log_event("yawn")
                        self.yawn_active = True
                else:
                    self.yawn_active = False

                # --- Overlay info ---
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Yawns: {self.yawn_count}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Drowsy Events: {self.drowsy_events}", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                status_text = "No Face Detected"
                status_color = (0, 0, 255)

            cv2.putText(frame, status_text, (30, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

            cv2.imshow("Drowsiness Detection System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        session_duration = time.time() - session_start
        self.logger.log_summary(
            duration_sec=session_duration,
            drowsy_events=self.drowsy_events,
            yawn_count=self.yawn_count,
        )

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Drowsiness Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--ear-threshold", type=float, default=0.25, help="EAR threshold")
    parser.add_argument("--ear-frames", type=int, default=20, help="Consecutive frames for drowsy alert")
    parser.add_argument("--mar-threshold", type=float, default=0.6, help="MAR threshold for yawn detection")
    parser.add_argument("--no-calibration", action="store_true", help="Skip calibration step")
    args = parser.parse_args()

    detector = DrowsinessDetector(
        ear_threshold=args.ear_threshold,
        ear_consec_frames=args.ear_frames,
        mar_threshold=args.mar_threshold,
        camera_index=args.camera,
    )
    detector.run(use_calibration=not args.no_calibration)


if __name__ == "__main__":
    main()
