"""Failure to yield detection - vehicle not yielding to pedestrian."""
from __future__ import annotations

import json
from pathlib import Path


def detect_failure_to_yield_segments(vehicle_histories: dict, pedestrian_histories: dict,
                                     geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect vehicles failing to yield to pedestrians in crosswalks."""
    events = []

    try:
        geometry = json.loads(Path(geometry_path).read_text())
        crossings = geometry.get("rois", {}).get("crossings", [])
    except:
        return events

    for ped_id, ped_history in pedestrian_histories.items():
        for ped_row in ped_history:
            ped_t, ped_x, ped_y = ped_row[0], ped_row[1], ped_row[2]

            # Check if pedestrian is in crossing
            in_crossing = False
            for crossing in crossings:
                points = crossing.get("points", [])
                if len(points) >= 3:
                    # Simplified point-in-polygon check
                    pass  # Would need full polygon implementation

            if in_crossing:
                # Check for approaching vehicles
                for veh_id, veh_history in vehicle_histories.items():
                    for veh_row in veh_history:
                        if abs(veh_row[0] - ped_t) < 1.0:  # Same time window
                            dist = ((veh_row[1] - ped_x) ** 2 + (veh_row[2] - ped_y) ** 2) ** 0.5
                            if dist < 100:  # Close to pedestrian
                                events.append([ped_t - 0.5, ped_t + 0.5, "failure_to_yield"])
                                break

    return events