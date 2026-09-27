"""Accident detection from bounding box contact and closing trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def detect_accident_candidates(histories: dict, geometry_path: str = "data/camera_geometry.json") -> tuple[list, list]:
    """Find accident candidates from overlapping vehicle trajectories.

    Returns:
        events: List of [start_sec, end_sec, "accident"]
        evidence: List of dicts with contact evidence
    """
    candidates = []
    evidence = []
    track_ids = list(histories.keys())

    for i, tid1 in enumerate(track_ids):
        for tid2 in track_ids[i + 1:]:
            h1, h2 = histories[tid1], histories[tid2]
            if len(h1) < 2 or len(h2) < 2:
                continue

            # Check each paired timestamp
            for idx1 in range(1, len(h1)):
                for idx2 in range(1, len(h2)):
                    t1_prev, t1_curr = h1[idx1 - 1][0], h1[idx1][0]
                    t2_prev, t2_curr = h2[idx2 - 1][0], h2[idx2][0]

                    # Skip if timestamps don't overlap
                    if max(t1_prev, t2_prev) >= min(t1_curr, t2_curr):
                        continue

                    # Get positions
                    x1a, y1a = h1[idx1 - 1][1], h1[idx1 - 1][2]
                    x1b, y1b = h1[idx1][1], h1[idx1][2]
                    x2a, y2a = h2[idx2 - 1][1], h2[idx2 - 1][2]
                    x2b, y2b = h2[idx2][1], h2[idx2][2]

                    # Interpolate positions at shared timestamp
                    shared_t = max(t1_prev, t2_prev)
                    if t1_curr != t1_prev:
                        f1 = (shared_t - t1_prev) / (t1_curr - t1_prev)
                    else:
                        f1 = 0
                    if t2_curr != t2_prev:
                        f2 = (shared_t - t2_prev) / (t2_curr - t2_prev)
                    else:
                        f2 = 0

                    x1 = x1a + f1 * (x1b - x1a)
                    y1 = y1a + f1 * (y1b - y1a)
                    x2 = x2a + f2 * (x2b - x2a)
                    y2 = y2a + f2 * (y2b - y2a)

                    # Calculate centers and estimate sizes
                    cx1, cy1 = x1, y1
                    cx2, cy2 = x2, y2

                    # Estimate box sizes from trajectory history
                    h1_sizes = [row[3] for row in h1 if row[3] > 0]
                    h2_sizes = [row[3] for row in h2 if row[3] > 0]
                    size1 = np.median(h1_sizes) if h1_sizes else 50
                    size2 = np.median(h2_sizes) if h2_sizes else 50

                    # Check for bounding box overlap
                    half1, half2 = size1 / 2, size2 / 2
                    overlap_x = max(0, min(cx1 + half1, cx2 + half2) - max(cx1 - half1, cx2 - half2))
                    overlap_y = max(0, min(cy1 + half1, cy2 + half2) - max(cy1 - half1, cy2 - half2))

                    if overlap_x > 0 and overlap_y > 0:
                        # Check closing trajectory
                        dx1 = x1b - x1a
                        dy1 = y1b - y1a
                        dx2 = x2b - x2a
                        dy2 = y2b - y2a

                        # Relative velocity
                        rel_dx = dx2 - dx1
                        rel_dy = dy2 - dy1

                        # Vector between centers
                        dist_dx = cx2 - cx1
                        dist_dy = cy2 - cy1

                        # Closing speed
                        closing_speed = (rel_dx * dist_dx + rel_dy * dist_dy) / max(np.hypot(dist_dx, dist_dy), 1)

                        # Check for post-contact speed drop
                        prev_speed1 = np.hypot(dx1, dy1) / max(t1_curr - t1_prev, 0.01)
                        curr_speed1 = np.hypot(x1 - x1a, y1 - y1a) / max(shared_t - t1_prev, 0.01) if idx1 > 0 else prev_speed1

                        if closing_speed > 5 or curr_speed1 < max(8, 0.1 * size1):
                            # High confidence accident: overlap + closing + stop
                            candidates.append([max(0, shared_t - 0.25), shared_t + 0.25, "accident"])
                            evidence.append({
                                "track_1": tid1,
                                "track_2": tid2,
                                "time": shared_t,
                                "closing_speed": closing_speed,
                                "overlap_area": overlap_x * overlap_y,
                            })

    # Merge overlapping candidates
    candidates.sort(key=lambda x: (x[2], x[0]))
    merged = []
    for start, end, label in candidates:
        if merged and merged[-1][2] == label and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, label])

    # Filter short events
    filtered = [e for e in merged if e[1] - e[0] >= 0.5]

    return filtered, evidence