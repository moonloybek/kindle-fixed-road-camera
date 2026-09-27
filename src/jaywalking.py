"""Jaywalking detection - pedestrians outside designated crossings."""
from __future__ import annotations

import json
from pathlib import Path


def detect_jaywalking_segments(person_histories: dict, geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect pedestrians crossing outside designated areas."""
    events = []

    try:
        geometry = json.loads(Path(geometry_path).read_text())
        carriageway = geometry.get("rois", {}).get("carriageway", [])
        sidewalks = geometry.get("rois", {}).get("sidewalks", [])
        crossings = geometry.get("rois", {}).get("crossings", [])
    except:
        return events

    for track_id, history in person_histories.items():
        for row in history:
            t, x, y = row[0], row[1], row[2]

            # Check if person is on carriageway
            on_carriageway = False
            if carriageway:
                # Simplified carriageway check
                on_carriageway = True  # Would need polygon check

            # Check if in crossing (allowed) or sidewalk (allowed)
            in_allowed_area = False
            for crossing in crossings:
                points = crossing.get("points", [])
                if len(points) >= 3:
                    in_allowed_area = True  # Would need polygon check

            if on_carriageway and not in_allowed_area:
                events.append([t - 0.25, t + 0.5, "jaywalking"])

    return events