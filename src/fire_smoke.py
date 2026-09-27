"""Lightweight frame appearance scores and temporal fire/smoke event grouping."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

SAMPLE_INTERVAL_SEC = 0.5
MIN_EVENT_SEC = 1.0
MAX_GAP_SEC = 1.0
SMOOTH_WINDOW = 3
MIN_SMOOTHED_SAMPLES = 3
MIN_SMOKE_COMPONENT_AREA_PX = 1000


class FireSmokeAnalyzer:
    def __init__(self, image_size=(640, 360)):
        self.image_size = image_size
        self.previous_gray = None
        self.previous_excluded = None
        self.samples = []

    def update(self, frame, timestamp, exclusion_boxes=()):
        """Score one sampled BGR frame; keep no full-resolution frame history."""
        small = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        excluded = np.zeros(gray.shape, dtype=np.uint8)
        sx, sy = self.image_size[0] / frame.shape[1], self.image_size[1] / frame.shape[0]
        for box in exclusion_boxes:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            margin_x = max(8, int((x2-x1)*sx*0.25))
            margin_y = max(8, int((y2-y1)*sy*0.25))
            xa = max(0, int(x1*sx)-margin_x)
            ya = max(0, int(y1*sy)-margin_y)
            xb = min(self.image_size[0], int(x2*sx)+margin_x)
            yb = min(self.image_size[1], int(y2*sy)+margin_y)
            if xb > xa and yb > ya:
                cv2.rectangle(excluded, (xa, ya), (xb, yb), 255, thickness=-1)

        hue, saturation, value = cv2.split(hsv)
        flame_color = (((hue <= 35) | (hue >= 170)) & (saturation >= 100) & (value >= 155)).astype(np.uint8) * 255
        flame_color[excluded > 0] = 0
        flame_color = cv2.morphologyEx(flame_color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fire_area = 0
        fire_hot_core = 0
        contours, _ = cv2.findContours(flame_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area < 350 or w < 8 or h < 12:
                continue
            x0, y0 = max(0, x), max(0, y)
            roi_h, roi_s, roi_v = hue[y0:y0+h, x0:x0+w], saturation[y0:y0+h, x0:x0+w], value[y0:y0+h, x0:x0+w]
            hot = int(np.count_nonzero((roi_h <= 32) & (roi_s >= 80) & (roi_v >= 210)))
            if hot >= 3:
                fire_area = max(fire_area, int(area))
                fire_hot_core = max(fire_hot_core, hot)

        smoke_area = 0
        smoke_upward_fraction = 0.0
        smoke_mean_upward_flow = 0.0
        if self.previous_gray is not None:
            difference = cv2.absdiff(gray, self.previous_gray)
            motion_excluded = excluded.copy()
            if self.previous_excluded is not None:
                motion_excluded = cv2.bitwise_or(motion_excluded, self.previous_excluded)
            motion_excluded = cv2.dilate(motion_excluded, np.ones((31, 31), np.uint8), iterations=1)
            difference[motion_excluded > 0] = 0
            smoke_mask = ((saturation <= 55) & (value >= 65) & (value <= 238) & (difference >= 9)).astype(np.uint8) * 255
            smoke_mask[motion_excluded > 0] = 0
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            flow = cv2.calcOpticalFlowFarneback(self.previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / max(float(h), 1.0)
                if area < MIN_SMOKE_COMPONENT_AREA_PX or h < 40 or w < 20 or not 0.25 <= aspect <= 2.0 or area > 0.25 * self.image_size[0] * self.image_size[1]:
                    continue
                region = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(region, [contour], -1, 255, thickness=-1)
                vertical_flow = flow[:, :, 1][region > 0]
                if len(vertical_flow) == 0:
                    continue
                upward = float(np.mean(vertical_flow <= -0.25))
                mean_vertical = float(np.mean(vertical_flow))
                if upward >= 0.65 and -6.0 <= mean_vertical <= -1.5:
                    smoke_area = max(smoke_area, int(area))
                    smoke_upward_fraction = max(smoke_upward_fraction, upward)
                    smoke_mean_upward_flow = min(smoke_mean_upward_flow, mean_vertical)

        self.previous_gray = gray
        self.previous_excluded = excluded
        sample = {
            "t_sec": float(timestamp),
            "fire_area_px_small": fire_area,
            "fire_hot_core_px_small": fire_hot_core,
            "smoke_motion_area_px_small": smoke_area,
            "smoke_upward_fraction": round(smoke_upward_fraction, 4),
            "smoke_mean_upward_flow_px_sample": round(smoke_mean_upward_flow, 3),
            "fire_raw": fire_area > 0,
            "smoke_raw": smoke_area > 0,
        }
        self.samples.append(sample)
        return sample

    def segments(self):
        """Median-smooth frame flags and emit time segments plus score evidence."""
        if not self.samples:
            return [], []
        n = len(self.samples)
        active = []
        for i in range(n):
            lo, hi = max(0, i - 1), min(n, i + 2)
            fire = sum(row["fire_raw"] for row in self.samples[lo:hi]) >= 2
            smoke = sum(row["smoke_raw"] for row in self.samples[lo:hi]) >= 2
            active.append((bool(fire), bool(smoke)))

        events, evidence = [], []
        for kind_index, kind in enumerate(("fire", "smoke")):
            indices = [i for i, flags in enumerate(active) if flags[kind_index]]
            if not indices:
                continue
            groups, group = [], [indices[0]]
            for index in indices[1:]:
                if self.samples[index]["t_sec"] - self.samples[group[-1]]["t_sec"] <= MAX_GAP_SEC + SAMPLE_INTERVAL_SEC:
                    group.append(index)
                else:
                    groups.append(group)
                    group = [index]
            groups.append(group)
            for group in groups:
                rows = [self.samples[i] for i in group]
                observed_duration = rows[-1]["t_sec"] - rows[0]["t_sec"]
                raw_count = sum(row[f"{kind}_raw"] for row in rows)
                if observed_duration < MIN_EVENT_SEC or raw_count < MIN_SMOOTHED_SAMPLES:
                    continue
                start = max(0.0, rows[0]["t_sec"] - SAMPLE_INTERVAL_SEC / 2)
                end = rows[-1]["t_sec"] + SAMPLE_INTERVAL_SEC / 2
                if end <= start:
                    continue
                events.append([float(start), float(end), "fire_smoke"])
                evidence.append({
                    "subtype": kind,
                    "start_sec": round(float(start), 3),
                    "end_sec": round(float(end), 3),
                    "evidence_samples": int(raw_count),
                    "peak_fire_area_px_small": max(r["fire_area_px_small"] for r in rows),
                    "peak_fire_hot_core_px_small": max(r["fire_hot_core_px_small"] for r in rows),
                    "peak_smoke_motion_area_px_small": max(r["smoke_motion_area_px_small"] for r in rows),
                    "peak_smoke_upward_fraction": max(r["smoke_upward_fraction"] for r in rows),
                    "strongest_smoke_upward_flow_px_sample": min(r["smoke_mean_upward_flow_px_sample"] for r in rows),
                })
        events.sort(key=lambda e: e[0])
        merged = []
        for start, end, label in events:
            if merged and start <= merged[-1][1] + SAMPLE_INTERVAL_SEC:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end, label])
        return merged, evidence