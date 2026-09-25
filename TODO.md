# Data TODO

## Prepare data

- [ ] Gather representative traffic videos and ground-truth event labels in the format shown in `examples/ground_truth.json`.
  - [x] Inventory the available clip and create an annotation sheet (`data/README.md`, `data/annotations.csv`).
  - [x] Screen the full clip at half-second intervals; no target events were confidently identified.
  - [x] Obtain user confirmation that `sample/C3905.MP4` contains no target events and record it in `data/ground_truth.json`.
  - [ ] Gather additional videos with verified labels across the event classes to be supported.
- [ ] Include examples for each event class the solution intends to predict; record video duration and frame rate.
- [ ] Split labeled videos into development and holdout sets so tuning does not use every example.

## Choose and set up detection

- [ ] Select a video object detector that can run within the competition's offline, GPU, time, and model-size limits.
- [ ] Document how model weights are stored or downloaded before the offline run.
- [ ] Add needed packages to `requirements.txt` and keep setup reproducible.
- [ ] Implement object detection and tracking helpers without changing the organizer-managed `run_submission.py` or `evaluate.py`.

## Implement event detection

- [ ] Start with a small set of observable classes, such as accidents and stopped vehicles, rather than emitting guesses for all classes.
- [ ] Convert detections and tracks into event candidates in `solution.py`.
- [ ] Merge short gaps, remove brief blips, and ensure same-class event segments do not overlap.
- [ ] Keep event times within the video duration and labels within `CLASSES`.

## Evaluate and improve

- [ ] Generate predictions on the development videos with `run_submission.py`.
- [ ] Check output format with `python evaluate.py --pred predictions.json --validate-only`.
- [ ] Score against labels with `python evaluate.py --pred predictions.json --gt my_labels.json --per-video`.
- [ ] Review false positives and missed events; tune thresholds using development data and track results on the holdout set.
- [ ] Measure runtime per video and keep within the harness budget.

## Optional risk estimation

- [ ] After event detection is stable, implement causal accident-risk scoring in `RiskEstimator.step` using only current and earlier frames.
- [ ] Evaluate accident anticipation separately and calibrate scores against labeled examples.
