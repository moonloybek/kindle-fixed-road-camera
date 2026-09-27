# WIUT Hackathon 2026 — Fixed Road Camera Traffic Event Detection

A computer vision system for detecting 14 types of traffic events from fixed-camera intersection video footage.

## Event Classes

| Category | Events |
|----------|--------|
| Accidents | `accident`, `near_miss` |
| Violations | `red_light`, `stop_line`, `illegal_turn`, `illegal_u_turn`, `solid_line_crossing`, `failure_to_yield` |
| Pedestrian | `jaywalking` |
| Vehicle | `wrong_way`, `stopped_vehicle`, `congestion`, `road_obstacle`, `fire_smoke` |

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

## Usage

```bash
# Generate predictions
python run_submission.py --videos sample --out predictions.json --team YOUR_TEAM

# Validate format
python evaluate.py --pred predictions.json --validate-only

# Evaluate against ground truth
python evaluate.py --pred predictions.json --gt my_labels.json --per-video
```

## Scoring

- **Part A:** F1@0.3, F1@0.5, F1@0.7 across all event classes
- **Part B:** Accident anticipation (AP, F1_alarm, mTTA)
- **Final:** 0.7 × Score_A + 0.3 × Score_B

## Runtime Constraint

Part A + Part B ≤ 3× video duration

## Repository

- GitHub: https://github.com/moonloybek/kindle-fixed-road-camera

## Team

- **Team:** Kindle
- **Role:** Computer Vision Track