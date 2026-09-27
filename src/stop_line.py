"""Stop line violation detection."""
from __future__ import annotations

import json
from pathlib import Path


def detect_stop_line_segments(histories: dict, signal_states: dict, geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect vehicles stopping beyond stop line on red."""
    events = []

    try:
        geometry = json.loads(Path(geometry_path).read_text())
        stop_lines = geometry.get("stop_lines", [])
    except:
        return events

    for track_id, history in histories.items():
        for i in range(len(history)):
            t = history[i][0]
            y = history[i][2]

            # Check if signal was red
            was_red = False
            for signal_id, states in signal_states.items():
                for ts, is_red in reversed(states):
                    if ts <= t:
                        was_red = is_red
                        break
                if was_red:
                    break

            if was_red:
                # Check if beyond stop line
                for sl in stop_lines:
                    if y > sl.get("y_threshold", 300):  # Beyond stop line
                        events.append([t, t + 1.0, "stop_line"])
                        break

    return events