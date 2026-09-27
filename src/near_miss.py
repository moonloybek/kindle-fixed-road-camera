"""Near-miss detection from close predicted trajectories without contact."""
from __future__ import annotations

import numpy as np


def detect_near_miss_candidates(histories: dict) -> tuple[list, list]:
    """Detect near-miss events fromTTC and evasive maneuvers.

    Returns:
        events: List of [start_sec, end_sec, "near_miss"]
        evidence: List of dicts with TTC and maneuver evidence
    """
    candidates = []
    evidence = []
    track_ids = list(histories.keys())
    MIN_TTC = 1.0  # seconds
    MANEUVER_WINDOW = 0.5  # seconds

    for i, tid1 in enumerate(track_ids):
        for tid2 in track_ids[i + 1:]:
            h1, h2 = histories[tid1], histories[tid2]
            if len(h1) < 3 or len(h2) < 3:
                continue

            for idx1 in range(2, len(h1)):
                for idx2 in range(2, len(h2)):
                    # Get positions and velocities
                    t1 = h1[idx1][0]
                    t2 = h2[idx2][0]

                    if abs(t1 - t2) > 0.5:
                        continue

                    x1, y1 = h1[idx1][1], h1[idx1][2]
                    x2, y2 = h2[idx2][1], h2[idx2][2]

                    # Previous positions for velocity
                    x1_prev, y1_prev = h1[idx1 - 1][1], h1[idx1 - 1][2]
                    x2_prev, y2_prev = h2[idx2 - 1][1], h2[idx2 - 1][2]

                    dt = max(t1 - h1[idx1 - 1][0], 0.01)
                    vx1, vy1 = (x1 - x1_prev) / dt, (y1 - y1_prev) / dt
                    vx2, vy2 = (x2 - x2_prev) / dt, (y2 - y2_prev) / dt

                    # Relative position and velocity
                    rel_x, rel_y = x2 - x1, y2 - y1
                    rel_vx, rel_vy = vx2 - vx1, vy2 - vy1

                    # Distance
                    distance = np.hypot(rel_x, rel_y)

                    # Time to closest approach (TTC)
                    if distance > 0:
                        closing_rate = (rel_x * rel_vx + rel_y * rel_vy) / distance
                        if closing_rate > 5:  # Approaching
                            ttc = distance / closing_rate
                        else:
                            ttc = float("inf")
                    else:
                        ttc = 0.0

                    # Check for evasive maneuver (sharp braking or heading change)
                    if idx1 >= 3:
                        # Compare recent velocity with earlier velocity
                        x1_earlier, y1_earlier = h1[idx1 - 2][1], h1[idx1 - 2][2]
                        vx1_earlier, vy1_earlier = (x1_prev - x1_earlier) / dt, (y1_prev - y1_earlier) / dt

                        braking = np.hypot(vx1 - vx1_earlier, vy1 - vy1_earlier)

                        # Heading change
                        heading_now = np.arctan2(vy1, vx1) if np.hypot(vx1, vy1) > 1 else 0
                        heading_prev = np.arctan2(vy1_earlier, vx1_earlier) if np.hypot(vx1_earlier, vy1_earlier) > 1 else 0
                        heading_change = abs(heading_now - heading_prev)
                        if heading_change > np.pi:
                            heading_change = 2 * np.pi - heading_change

                        maneuver = braking > 20 or heading_change > 0.7  # ~40 degrees

                        if 0 < ttc <= MIN_TTC and maneuver:
                            # Check no contact around maneuver
                            size1 = h1[idx1][3] if len(h1[idx1]) > 3 else 50
                            if distance > size1 * 0.5:  # No contact
                                candidates.append([t1 - 0.25, t1 + 0.5, "near_miss"])
                                evidence.append({
                                    "track_1": tid1,
                                    "track_2": tid2,
                                    "time": t1,
                                    "ttc": round(ttc, 3),
                                    "clearance": round(distance, 1),
                                    "braking": round(braking, 1),
                                    "heading_change": round(heading_change, 3),
                                })

    # Merge overlapping
    candidates.sort(key=lambda x: (x[2], x[0]))
    merged = []
    for start, end, label in candidates:
        if merged and merged[-1][2] == label and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, label])

    return [e for e in merged if e[1] - e[0] >= 0.5], evidence