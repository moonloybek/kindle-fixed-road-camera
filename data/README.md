# Data Preparation

## Available video

| File | Duration | FPS | Resolution | Review status |
| --- | ---: | ---: | --- | --- |
| `sample/C3905.MP4` | 127.6 s | 29.97 | 3840x2160 | Human-confirmed: no target events |

The clip was visually sampled across its full duration at half-second intervals. No clearly verifiable accident, near miss, or other labeled event was apparent. A slow-moving truck occupies the junction around 60–105 s, but it continues moving, so it was not labeled as a stopped vehicle. The user confirmed that the full clip contains no target events. The reviewed annotation is recorded in `annotations.csv` and `ground_truth.json`.

The JSON files in `examples/` demonstrate the expected format, but their video IDs do not match the available clip. Do not use them as labels for `C3905.MP4`.

## Annotation workflow

1. Review candidate intervals frame by frame before assigning an event label and precise boundaries.
2. Add one row per confirmed event to `annotations.csv`, using seconds from the first frame and a label from the official class list in `solution.py`.
3. If confirming there are no target events, mark the video reviewed and record that explicit status in the notes.
4. Keep `ground_truth.json` synchronized with reviewed annotations before scoring.

Add more representative videos to the inventory as they become available. Keep source videos out of Git if they are large or restricted; record their paths and access instructions here instead.
