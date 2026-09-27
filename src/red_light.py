"""Red light violation detection."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_signal_rois(geometry_path: str = "data/camera_geometry.json") -> dict:
    """Load signal lamp ROIs from geometry."""
    try:
        geometry = json.loads(Path(geometry_path).read_text())
        signals = geometry.get("signal_lamps", [])
        return {s["id"]: s["bbox"] for s in signals}
    except:
        return {}


def red_lamp_visible(frame: np.ndarray, bbox: tuple) -> bool:
    """Detect if red lamp is visible in ROI."""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Red color range (handles wraparound at hue=0)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    return red_pixels > 0.05 * total_pixels


def detect_red_light_segments(histories: dict, signal_states: dict, geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect red light violations."""
    events = []

    try:
        geometry = json.loads(Path(geometry_path).read_text())
        stop_lines = geometry.get("stop_lines", [])
    except:
        return events

    for track_id, history in histories.items():
        for i in range(1, len(history)):
            t = history[i][0]

            # Check if signal was red at this time
            was_red = False
            for signal_id, states in signal_states.items():
                for ts, is_red in reversed(states):
                    if ts <= t:
                        was_red = is_red
                        break
                if was_red:
                    break

            if was_red:
                # Check if vehicle crossed stop line
                for stop_line in stop_lines:
                    # Simplified: check if vehicle is in intersection area
                    if history[i][2] > stop_line.get("y_threshold", 300):
                        events.append([t - 0.25, t + 0.5, "red_light"])
                        break

    return events