"""
SessionLogger: records drowsiness/yawn events with timestamps and writes
a session summary to a CSV log file for post-session analysis.
"""

import csv
import os
import time
from datetime import datetime


class SessionLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_log_path = os.path.join(log_dir, f"events_{timestamp}.csv")
        self.summary_log_path = os.path.join(log_dir, "session_summary.csv")

        with open(self.event_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type"])

    def log_event(self, event_type: str):
        with open(self.event_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), event_type])

    def log_summary(self, duration_sec: float, drowsy_events: int, yawn_count: int):
        file_exists = os.path.isfile(self.summary_log_path)
        with open(self.summary_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["session_end_time", "duration_sec", "drowsy_events", "yawn_count"]
                )
            writer.writerow(
                [datetime.now().isoformat(), round(duration_sec, 2), drowsy_events, yawn_count]
            )
        print(f"[INFO] Session summary saved to {self.summary_log_path}")
        print(f"[INFO] Duration: {duration_sec:.1f}s | Drowsy Events: {drowsy_events} | Yawns: {yawn_count}")
