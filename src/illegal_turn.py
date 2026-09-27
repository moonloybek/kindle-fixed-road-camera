"""Illegal turn detection from prohibited movements."""
from __future__ import annotations

import json
from pathlib import Path


def detect_illegal_turn_segments(histories: dict, geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect illegal turns based on intersection geometry."""
    events = []

    try:
        geometry = json.loads(Path(geometry_path).read_text())
        turn_permissions = geometry.get("turn_permissions", {})
    except:
        return events

    for track_id, history in histories.items():
        if len(history) < 5:
            continue

        # Simplified: detect sharp turns in intersection area
        x, y = history[-1][1], history[-1][2]
        dx = history[-1][1] - history[0][1]
        dy = history[-1][2] - history[0][2]

        # Check for sharp turn (90+ degrees)
        if abs(dx) > 20 and abs(dy) > 20:
            # This is a simplified check - actual implementation needs verified turn rules
            pass  # Would need verified turn permissions to label as illegal

    return events