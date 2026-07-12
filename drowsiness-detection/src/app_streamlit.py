"""
Optional Streamlit dashboard for the Drowsiness Detection System.
Run with: streamlit run src/app_streamlit.py
"""

import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

from utils import eye_aspect_ratio, mouth_aspect_ratio, LEFT_EYE, RIGHT_EYE, MOUTH, play_alert_sound
from logger import SessionLogger

st.set_page_config(page_title="Drowsiness Detection Dashboard", layout="wide")
st.title("🚗 Real-Time Drowsiness Detection Dashboard")

col1, col2 = st.columns([3, 1])
frame_placeholder = col1.empty()

with col2:
    st.subheader("Session Stats")
    ear_metric = st.empty()
    mar_metric = st.empty()
    yawn_metric = st.empty()
    drowsy_metric = st.empty()
    status_placeholder = st.empty()

ear_threshold = st.sidebar.slider("EAR Threshold", 0.10, 0.40, 0.25, 0.01)
mar_threshold = st.sidebar.slider("MAR Threshold (Yawn)", 0.30, 1.00, 0.60, 0.01)
ear_consec_frames = st.sidebar.slider("Consecutive Frames for Alert", 5, 40, 20, 1)
run = st.sidebar.checkbox("Start Camera", value=False)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

logger = SessionLogger()

if run:
    cap = cv2.VideoCapture(0)
    eye_closed_frames = 0
    yawn_count = 0
    drowsy_events = 0
    alert_active = False

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape
        status = "Awake"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks, MOUTH, w, h)

            if avg_ear < ear_threshold:
                eye_closed_frames += 1
                if eye_closed_frames >= ear_consec_frames:
                    status = "DROWSY!"
                    if not alert_active:
                        drowsy_events += 1
                        logger.log_event("drowsy_eyes_closed")
                        play_alert_sound()
                        alert_active = True
            else:
                eye_closed_frames = 0
                alert_active = False

            if mar > mar_threshold:
                status = "YAWNING"
                yawn_count += 1
                logger.log_event("yawn")

            ear_metric.metric("EAR", f"{avg_ear:.2f}")
            mar_metric.metric("MAR", f"{mar:.2f}")
            yawn_metric.metric("Yawns", yawn_count)
            drowsy_metric.metric("Drowsy Events", drowsy_events)
            status_placeholder.markdown(f"### Status: **{status}**")

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(0.01)

    cap.release()
else:
    st.info("Check 'Start Camera' in the sidebar to begin detection.")
