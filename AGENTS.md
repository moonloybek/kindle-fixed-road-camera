# Repository Guidelines

## Project Structure

- `solution.py` is the team implementation. Keep the required `CLASSES`, `detect_events(video_path)`, and `RiskEstimator.reset/step` interface intact.
- `run_submission.py` runs the solution against video files and writes predictions; treat it as organizer-managed.
- `evaluate.py` validates prediction files and calculates the official metrics; treat it as organizer-managed.
- `examples/` contains sample prediction and ground-truth JSON files. `requirements.txt` lists the harness dependencies.
- Add helper modules, model code, or configuration in clearly named files or directories (for example, `src/` or `weights/`). Keep large model weights within the competition limit and document how they are obtained.

## Build, Test, and Development

Install the listed dependencies with `pip install -r requirements.txt`. Implement and iterate in `solution.py`, then run the harness on local videos:

```bash
python run_submission.py --videos samples --out predictions_samples.json --team your-team
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only
```

The first command generates predictions, the second scores them against your labels, and the third checks format without ground truth. The repository does not define a separate build step or automated test suite.

## Coding Style

Use Python conventions: four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Preserve the public names and signatures in `solution.py`; labels must come from the official class list. Keep event times in seconds, risk values in `[0, 1]`, and `RiskEstimator.step` causal (use only the current and earlier frames). Add dependencies only when needed and record them in `requirements.txt`.

## Testing Guidelines

There is no configured test framework or coverage threshold. Use the submission harness and evaluator as the primary checks. Test with representative videos and labels, and run `--validate-only` to catch malformed output. Confirm every input video appears in the prediction output, including videos with no events.

## Commits and Pull Requests

Use short, imperative commit subjects, such as `Improve event segment merging`. In pull requests, summarize the model or logic change, include the commands and data used for evaluation, report relevant metric/runtime changes, and link the related task or issue when available. Do not include private test data or credentials.
