"""Solid line crossing detection."""
from __future__ import annotations

import json
from pathlib import Path


def detect_solid_line_crossing_segments(histories: dict, geometry_path: str = "data/camera_geometry.json") -> list:
    """Detect vehicles crossing solid lane markings."""
    events = []

    # This requires verified solid line coordinates
    # Currently the geometry has no verified solid line segments
    # See outputs/SOLID_LINE_CROSSING_REVIEW.md

    return events