# Integration Notes

## Design principle

The audio component is a library plus an optional standalone Streamlit dashboard. Keep analysis functions independent from UI code, so the main application can call them from its own upload workflow or later through an API.

## Public service interface

`analyze_audio(data: bytes, filename: str, include_synthetic_screening: bool = False, model_id: str = DEFAULT_SYNTHETIC_MODEL_ID) -> dict`

- `data`: exact bytes received by the host upload control.
- `filename`: display name only; not trusted as content proof.
- synthetic screening is opt-in because it has a large model download and uncertain multilingual performance.
- function returns JSON-safe metadata and findings.

## Host application responsibilities

1. Assign the case ID and preserve the original upload in approved storage.
2. Keep the original fingerprint and upload metadata in the host's audit log.
3. Store analysis JSON and generated report as separate artifacts, linked by evidence ID.
4. Avoid writing any uploaded audio or report into the source repository.
5. Make the audio module's limitations visible in the UI and report.

## Finding contract

Candidate edit findings contain:
- `finding_id`
- `module = "audio"`
- `finding_type`
- `timestamp_start_seconds`, `timestamp_end_seconds`, `peak_timestamp_seconds`
- `score` plus `score_type` (relative heuristic anomaly score, not probability)
- `supporting_features`
- `explanation` and `limitations`

Synthetic-speech segments contain a model ID, start/end timestamps, model scores for the configured classes and model label. Preserve the model's documented class mapping; do not infer class order from a generic label name.

## Do not combine scores blindly

The signal anomaly score and the synthetic-classifier score have different meanings and scales. Do not average them into a universal authenticity score. Keep them as separate findings and let the investigator review the supporting evidence.
