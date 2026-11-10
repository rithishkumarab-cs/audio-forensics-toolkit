"""Lightweight ONNX audio anti-spoofing screening adapter.

Experimental screening only. A model score is not proof of authenticity
or deliberate manipulation.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .config import DEFAULT_SAMPLE_RATE, DEFAULT_SYNTHETIC_MODEL_ID, MODEL_MIN_WINDOW_SECONDS


WINDOW_SAMPLES = 64600
WINDOW_SECONDS = WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "aasist-l.onnx"
)


@lru_cache(maxsize=1)
def _load_session(model_path: str):
    """Load and validate the local ONNX model once per process."""
    path = Path(model_path)

    if not path.is_file():
        raise RuntimeError(
            f"AASIST-L ONNX model not found at {path}. "
            "Place aasist-l.onnx in the project's models directory."
        )

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            "Unexpected AASIST-L ONNX interface: expected one input and one output."
        )

    input_info = inputs[0]
    output_info = outputs[0]

    if input_info.shape[-1] != WINDOW_SAMPLES:
        raise RuntimeError(
            f"Unexpected model input shape: {input_info.shape}; "
            f"expected a final dimension of {WINDOW_SAMPLES}."
        )

    if output_info.shape[-1] != 2:
        raise RuntimeError(
            f"Unexpected model output shape: {output_info.shape}; "
            "expected two class logits."
        )

    return session, input_info.name, output_info.name


def _fixed_window(chunk: np.ndarray) -> np.ndarray:
    """Prepare the deterministic 64,600-sample model input."""
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)

    if chunk.size == 0:
        raise ValueError("Cannot prepare an empty audio window.")

    if chunk.size >= WINDOW_SAMPLES:
        return chunk[:WINDOW_SAMPLES]

    # Match the published wrapper's deterministic repeat-padding behaviour.
    repeats = (WINDOW_SAMPLES + chunk.size - 1) // chunk.size
    return np.tile(chunk, repeats)[:WINDOW_SAMPLES].astype(
        np.float32,
        copy=False,
    )


def screen_synthetic_speech(
    signal: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    model_id: str = DEFAULT_SYNTHETIC_MODEL_ID,
    window_seconds: float | None = None,
) -> dict[str, Any]:
    """Screen non-overlapping audio windows for spoof-like characteristics.

    The model outputs class 0 = spoof-like and class 1 = bona-fide-like.
    Softmax scores are model scores, not calibrated probabilities.
    """
    if model_id != DEFAULT_SYNTHETIC_MODEL_ID:
        raise ValueError(
            "Only the configured AASIST-L ONNX model is supported."
        )

    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"Audio must be resampled to {DEFAULT_SAMPLE_RATE} Hz first."
        )

    x = np.asarray(signal, dtype=np.float32).reshape(-1)

    if x.size == 0:
        raise ValueError("Audio signal is empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Audio signal contains NaN or infinite values.")

    min_samples = int(
        round(MODEL_MIN_WINDOW_SECONDS * sample_rate)
    )

    common_result = {
        "method": "aasist_l_onnx_anti_spoofing",
        "model_id": model_id,
        "model_path": str(MODEL_PATH),
        "sample_rate_hz": sample_rate,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": round(WINDOW_SECONDS, 4),
        "device": "CPUExecutionProvider",
        "model_license_reported": (
            "MIT according to the published repository; "
            "verify the model card before redistribution."
        ),
        "model_scope": (
            "Experimental audio anti-spoofing baseline trained on "
            "ASVspoof2019 Logical Access data."
        ),
    }

    if x.size < min_samples:
        return {
            **common_result,
            "status": "inconclusive",
            "message": (
                f"At least {MODEL_MIN_WINDOW_SECONDS:.1f} seconds of audio "
                "are required for screening."
            ),
            "segments": [],
            "limitations": [
                "Clip is shorter than the configured minimum window.",
                "No authenticity conclusion can be drawn.",
            ],
        }

    session, input_name, output_name = _load_session(
        str(MODEL_PATH.resolve())
    )

    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    segments: list[dict[str, Any]] = []
    processed_until = 0

    # Use non-overlapping windows to reduce redundant correlated scores.
    for start in range(0, x.size, WINDOW_SAMPLES):
        chunk = x[start:start + WINDOW_SAMPLES]

        # Do not report a decision on a very short final tail.
        if chunk.size < min_samples:
            break

        model_input = _fixed_window(chunk)[None, :]
        logits = session.run(
            [output_name],
            {input_name: model_input},
        )[0]

        logits = np.asarray(logits, dtype=np.float64).reshape(-1)

        if logits.size != 2 or not np.all(np.isfinite(logits)):
            raise RuntimeError(
                "AASIST-L returned invalid class scores."
            )

        # Numerically stable softmax.
        shifted = logits - np.max(logits)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores)

        # Published AASIST-L mapping:
        # index 0 = spoof, index 1 = bona fide.
        score_fake = float(probabilities[0])
        score_real = float(probabilities[1])

        end = min(start + chunk.size, x.size)
        padded = chunk.size < WINDOW_SAMPLES

        segments.append({
            "segment_id": f"AUD-SYN-{len(segments) + 1:03d}",
            "timestamp_start_seconds": round(start / sample_rate, 3),
            "timestamp_end_seconds": round(end / sample_rate, 3),
            "model_score_fake_class": round(score_fake, 6),
            "model_score_real_class": round(score_real, 6),
            "model_label": (
                "synthetic-like"
                if score_fake > score_real
                else "human-like"
            ),
            "score_interpretation": (
                "Softmax-derived model score, not a calibrated probability "
                "of deception or authenticity."
            ),
            "padded_by_repetition": padded,
        })

        processed_until = end

    return {
        **common_result,
        "model_sha256": model_hash,
        "status": "completed" if segments else "inconclusive",
        "segments": segments,
        "unscored_tail_seconds": round(
            max(0, x.size - processed_until) / sample_rate,
            3,
        ),
        "limitations": [
            "Synthetic-like and human-like are model classifications, not verdicts.",
            "Scores are not calibrated probabilities of deception.",
            "Performance can change with language, speaker, codec, noise, and unseen generators.",
            "The published benchmark reports substantially weaker performance on out-of-domain datasets.",
            "Validate on separate genuine and synthetic recordings before operational use.",
        ],
    }