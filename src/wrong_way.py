"""Wrong-way vehicle detection from opposing lane flow."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def detect_wrong_way_segments(histories: dict, geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect vehicles traveling opposite to normal traffic flow."""
    events = []

    try:
        geometry = json.loads(Path(geometry_path).read_text())
        lane_centerlines = geometry.get("lane_centerlines", [])
    except:
        lane_centerlines = []

    for track_id, history in histories.items():
        if len(history) < 8:  # Need at least 4 seconds of tracking at 2 FPS
            continue

        # Calculate trajectory direction
        dx = history[-1][1] - history[0][1]
        dy = history[-1][2] - history[0][2]

        if np.hypot(dx, dy) < 50:  # Not moving enough
            continue

        trajectory_angle = np.arctan2(dy, dx)

        # Check against lane flows
        opposing_count = 0
        for lane in lane_centerlines:
            if len(lane) >= 2:
                lane_dx = lane[1][0] - lane[0][0]
                lane_dy = lane[1][1] - lane[0][1]
                lane_angle = np.arctan2(lane_dy, lane_dx)

                angle_diff = abs(trajectory_angle - lane_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                if angle_diff > np.pi / 2:  # Opposing direction
                    opposing_count += 1

        if opposing_count >= 2:  # Sustained reverse travel
            # Find first opposing sample
            start_time = history[0][0]
            end_time = history[-1][0]
            events.append([start_time, end_time, "wrong_way"])

    return events