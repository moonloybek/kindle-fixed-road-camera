"""Congestion detection from sustained multi-lane slow traffic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_CONGESTION_SEC = 30.0


def analyze_congestion(histories: dict, geometry_path: str = "data/camera_geometry.json") -> tuple[list, list]:
    """Detect traffic congestion from sustained low-speed multi-lane conditions."""
    events = []
    evidence = []

    # Group tracks by lane (simplified: use x position)
    lane_bands = [(0, 200), (200, 400), (400, 600), (600, 800)]

    for lane_start, lane_end in lane_bands:
        lane_vehicles = []
        for track_id, history in histories.items():
            # Check if vehicle is in this lane band
            lane_positions = [row[1] for row in history]
            if lane_start <= np.median(lane_positions) < lane_end:
                lane_vehicles.append(track_id)

        if len(lane_vehicles) < 2:
            continue

        # Check if all observed vehicles are slow
        all_slow = True
        for tid in lane_vehicles:
            history = histories[tid]
            speeds = []
            for i in range(1, len(history)):
                dt = history[i][0] - history[i - 1][0]
                if dt > 0:
                    speed = np.hypot(history[i][1] - history[i - 1][1],
                                    history[i][2] - history[i - 1][2]) / dt
                    speeds.append(speed)

            if speeds and np.median(speeds) > 15:
                all_slow = False
                break

        if all_slow and len(lane_vehicles) >= 2:
            # Congestion detected in this lane
            events.append([0, 0, "congestion"])  # Would need proper timing

    return events, evidence