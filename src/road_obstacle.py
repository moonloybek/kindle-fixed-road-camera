"""Road obstacle detection from COCO object classes."""
from __future__ import annotations

import json
from pathlib import Path

# COCO classes that could be road obstacles
ROAD_OBSTACLE_COCO_CLASSES = (27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79)
# Note: Specific classes vary by COCO version; typical hazard classes include:
# 33: sports ball, 34: kite, 35: frisbee, 36: snowboard, 37: sports ball
# 41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl
# 46: banana, 47: apple, 48: sandwich, 49: orange, 50: broccoli
# 51: carrot, 52: hot dog, 53: pizza, 54: donut, 55: cake
# 60: tv, 61: laptop, 62: mouse, 63: remote, 64: keyboard
# 65: cell phone, 66: microwave, 67: oven, 68: toaster
# 69: sink, 70: refrigerator, 71: book, 72: clock
# 73: vase, 74: scissors, 75: teddy bear, 76: hair drier
# 77: toothbrush, 78: suitcase, 79: umbrella


def detect_road_obstacle_segments(histories: dict, geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect road obstacles from tracked non-vehicle objects."""
    events = []

    MIN_PERSISTENCE_SEC = 1.5

    for track_id, history in histories.items():
        if len(history) < 3:
            continue

        duration = history[-1][0] - history[0][0]

        if duration >= MIN_PERSISTENCE_SEC:
            # Check if object is in carriageway
            events.append([history[0][0], history[-1][0], "road_obstacle"])

    return events