"""Illegal U-turn detection from sharp heading changes."""
from __future__ import annotations

import json
from pathlib import Path


def detect_illegal_u_turn_segments(histories: dict, geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect U-turn maneuvers from heading changes > 135 degrees."""
    events = []

    MIN_HEADING_CHANGE = 2.356  # ~135 degrees in radians
    MIN_TRACK_LENGTH = 10  # samples at 2 FPS = 5 seconds

    for track_id, history in histories.items():
        if len(history) < MIN_TRACK_LENGTH:
            continue

        # Calculate heading at start and end
        dx_start = history[5][1] - history[0][1]
        dy_start = history[5][2] - history[0][2]
        dx_end = history[-1][1] - history[-6][1]
        dy_end = history[-1][2] - history[-6][2]

        if abs(dx_start) < 5 and abs(dy_start) < 5:
            continue  # Not moving at start

        heading_start = __class__.__safe_atan2(dy_start, dx_start)
        heading_end = __class__.__safe_atan2(dy_end, dx_end)

        heading_change = abs(heading_start - heading_end)
        if heading_change > 3.14159:
            heading_change = 2 * 3.14159 - heading_change

        if heading_change >= MIN_HEADING_CHANGE:
            # U-turn detected
            events.append([history[0][0], history[-1][0], "illegal_u_turn"])

    return events


def __safe_atan2(dy, dx):
    """Safe arctangent with small value handling."""
    import math
    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return 0
    return math.atan2(dy, dx)