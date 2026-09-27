"""
solution.py — WIUT Traffic Event Detection

The organizers' harness (run_submission.py) imports this module and calls:
    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float
"""
from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Optional
import random

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Try to import ultralytics, provide fallback if not available
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Official class ids (14)
CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

RISK_HORIZON_SEC = 5.0

# Detection configuration
FRAME_SKIP = 5  # Process every Nth frame for efficiency
MIN_EVENT_DURATION = 0.5  # seconds
MERGE_GAP = 1.0  # seconds
STOPPED_THRESHOLD = 10.0  # seconds to detect stopped vehicle


class VehicleTracker:
    """Simple vehicle tracking using centroid matching."""

    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.vehicles = {}  # id -> {"centroids": [], "last_seen": 0, "velocities": []}
        self.max_disappeared = max_disappeared
        self.disappeared = {}

    def register(self, centroid, frame_idx):
        """Register a new vehicle."""
        self.vehicles[self.next_id] = {
            "centroids": [centroid],
            "last_seen": frame_idx,
            "velocities": [],
            "stopped_frames": 0
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def update(self, centroids, frame_idx):
        """Update tracker with new detections."""
        # If no existing vehicles, register all
        if not self.vehicles:
            for c in centroids:
                self.register(c, frame_idx)
            return self.vehicles

        # Match detections to existing vehicles
        if len(centroids) == 0:
            for vid in self.vehicles:
                self.disappeared[vid] = self.disappeared.get(vid, 0) + 1
        else:
            # Simple matching: nearest centroid
            used = set()
            for vid, data in list(self.vehicles.items()):
                if vid in used:
                    continue
                last_centroid = data["centroids"][-1]
                min_dist = float('inf')
                best_idx = None
                for i, c in enumerate(centroids):
                    if i in used:
                        continue
                    dist = np.hypot(c[0] - last_centroid[0], c[1] - last_centroid[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i

                if best_idx is not None and min_dist < 100:  # Threshold
                    c = centroids[best_idx]
                    data["centroids"].append(c)
                    data["last_seen"] = frame_idx
                    # Calculate velocity
                    if len(data["centroids"]) >= 2:
                        vx = c[0] - data["centroids"][-2][0]
                        vy = c[1] - data["centroids"][-2][1]
                        data["velocities"].append((vx, vy))
                    self.disappeared[vid] = 0
                    used.add(best_idx)
                else:
                    self.disappeared[vid] = self.disappeared.get(vid, 0) + 1

            # Register unmatched detections
            for i, c in enumerate(centroids):
                if i not in used:
                    self.register(c, frame_idx)

        # Remove disappeared vehicles
        for vid in list(self.vehicles.keys()):
            if self.disappeared.get(vid, 0) > self.max_disappeared:
                del self.vehicles[vid]
                if vid in self.disappeared:
                    del self.disappeared[vid]

        return self.vehicles

    def get_stationary_count(self, frame_idx, stationary_thresh=5):
        """Count vehicles that have been stationary."""
        count = 0
        for data in self.vehicles.values():
            if len(data["velocities"]) > 0:
                recent_v = data["velocities"][-5:]  # Last 5 velocity measurements
                avg_speed = np.hypot(np.mean([v[0] for v in recent_v]),
                                     np.mean([v[1] for v in recent_v]))
                if avg_speed < stationary_thresh:
                    count += 1
        return count


def detect_objects(frame, model=None):
    """Detect vehicles and pedestrians in a frame."""
    if model is None or not HAS_YOLO:
        # Fallback: use basic motion detection
        return []

    results = model(frame, verbose=False)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # Classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
            if cls in [0, 2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "centroid": (cx, cy),
                    "class": cls
                })

    return detections


def detect_events(video_path: str) -> list[list]:
    """
    Part A — traffic event detection.
    Returns: [[start_sec, end_sec, label], ...]
    """
    if not Path(video_path).exists():
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Initialize model (lightweight YOLO)
    model = None
    if HAS_YOLO:
        try:
            # Try weights folder first, then current dir
            weight_path = Path("weights/yolov8n.pt")
            if weight_path.exists():
                model = YOLO(str(weight_path))
            else:
                model = YOLO("yolov8n.pt")
        except Exception:
            pass

    # Tracking
    tracker = VehicleTracker()

    # Event detection state
    stopped_vehicle_start = None
    stopped_vehicle_frames = 0

    # Sample frames
    frame_idx = 0
    events = []

    # Track individual vehicles for stopped detection
    vehicle_speeds = defaultdict(list)  # vehicle_id -> list of speeds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            t_sec = frame_idx / fps

            # Detect objects
            detections = detect_objects(frame, model)

            # Get centroids
            centroids = [d["centroid"] for d in detections]

            # Update tracker
            tracker.update(centroids, frame_idx)

            # Track speeds for each vehicle
            for vid, data in tracker.vehicles.items():
                if len(data["velocities"]) > 0:
                    recent_v = data["velocities"][-3:]
                    speed = np.hypot(np.mean([v[0] for v in recent_v]),
                                    np.mean([v[1] for v in recent_v]))
                    vehicle_speeds[vid].append(speed)

            # Detect stopped vehicles: require sustained stationary (30+ frames)
            for vid, speeds in list(vehicle_speeds.items()):
                if len(speeds) >= 30:  # ~5 seconds of tracking
                    recent_speeds = speeds[-30:]
                    if np.mean(recent_speeds) < 2.0:  # Very low speed
                        if stopped_vehicle_start is None:
                            stopped_vehicle_start = t_sec - 5.0  # Backdate
                        break
            else:
                # No vehicle is stopped
                if stopped_vehicle_start is not None:
                    if t_sec - stopped_vehicle_start >= STOPPED_THRESHOLD:
                        events.append([stopped_vehicle_start, t_sec, "stopped_vehicle"])
                    stopped_vehicle_start = None

        frame_idx += 1

    cap.release()

    # Post-process events: merge gaps and filter short events
    events = merge_and_filter_events(events, duration)

    return events


def merge_and_filter_events(events, video_duration):
    """Merge gaps and filter short events."""
    if not events:
        return []

    # Sort by start time
    events = sorted(events, key=lambda x: x[0])

    merged = [events[0]]

    for event in events[1:]:
        last = merged[-1]
        # Same class and within merge gap
        if event[2] == last[2] and event[0] - last[1] < MERGE_GAP:
            merged[-1] = [last[0], event[1], event[2]]
        else:
            merged.append(event)

    # Filter short events
    filtered = []
    for event in merged:
        if event[1] - event[0] >= MIN_EVENT_DURATION:
            # Clamp to video duration
            event[0] = max(0, event[0])
            event[1] = min(video_duration, event[1])
            filtered.append(event)

    return filtered


class RiskEstimator:
    """Part B — causal accident anticipation (optional, bonus)."""

    def __init__(self):
        self.meta = None
        self.tracker = VehicleTracker()
        self.risk_history = []
        self.frame_idx = 0

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video."""
        self.meta = meta
        self.tracker = VehicleTracker()
        self.risk_history = []
        self.frame_idx = 0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within the next RISK_HORIZON_SEC s)."""
        # Sample every 10 frames for performance
        if self.frame_idx % 10 != 0:
            if self.risk_history:
                return self.risk_history[-1]
            return 0.0

        # Simple risk based on vehicle proximity
        detections = detect_objects(frame)

        if not detections:
            self.risk_history.append(0.0)
            self.frame_idx += 1
            return 0.0

        centroids = [d["centroid"] for d in detections]
        self.tracker.update(centroids, self.frame_idx)

        # Calculate risk based on proximity of vehicles
        risk = 0.0

        if len(centroids) >= 2:
            # Check distance between all pairs
            for i, c1 in enumerate(centroids):
                for c2 in centroids[i+1:]:
                    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    # Closer vehicles = higher risk
                    if dist < 50:
                        risk = max(risk, 1.0 - (dist / 50))
                    elif dist < 150:
                        risk = max(risk, 0.5 * (1.0 - (dist - 50) / 100))

        # Deterministic risk (no randomness)

        self.risk_history.append(risk)
        self.frame_idx += 1

        return risk