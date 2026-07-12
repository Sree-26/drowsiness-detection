"""
Utility functions: landmark indices, EAR/MAR calculations, and alert sound.
"""

import math
import threading

# MediaPipe Face Mesh landmark indices for eyes and mouth
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]


def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _landmark_to_px(landmark, w, h):
    return (landmark.x * w, landmark.y * h)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Compute Eye Aspect Ratio (EAR) given 6 landmark indices for one eye.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = [_landmark_to_px(landmarks[i], w, h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts

    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    """
    Compute Mouth Aspect Ratio (MAR) for yawn detection.
    Uses vertical mouth opening distances relative to mouth width.
    """
    pts = [_landmark_to_px(landmarks[i], w, h) for i in mouth_indices]
    p_left, p_top1, p_top2, p_right, p_bottom1, p_bottom2, p_top_mid, p_bottom_mid = pts

    vertical = _euclidean(p_top_mid, p_bottom_mid)
    horizontal = _euclidean(p_left, p_right)

    if horizontal == 0:
        return 0.0

    mar = vertical / horizontal
    return mar


def play_alert_sound(sound_path: str = "assets/alert.wav"):
    """Play alert sound asynchronously so it doesn't block the video loop."""
    def _play():
        try:
            import winsound
            winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            print("\a")

    threading.Thread(target=_play, daemon=True).start()
