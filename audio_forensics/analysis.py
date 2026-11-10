"""Public service API for the audio forensics component."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any

from .audio_io import load_audio_bytes
from .config import DEFAULT_SYNTHETIC_MODEL_ID
from .edit_analysis import analyze_edit_candidates
from .synthetic_detector import screen_synthetic_speech


def analyze_decoded_audio(
    data: bytes,
    filename: str,
    signal,
    metadata: dict[str, Any],
    *,
    include_synthetic_screening: bool = False,
    model_id: str = DEFAULT_SYNTHETIC_MODEL_ID,
) -> dict[str, Any]:
    """Analyse a decoded working copy when the caller already performed ingestion.

    This avoids decoding twice in applications that need both the signal for plotting and
    the structured result. `data` is still used only for exact original-byte identity.
    """
    if metadata.get("sha256") is None:
        raise ValueError("metadata must include the SHA-256 of the original bytes")
    if metadata["sha256"] != hashlib.sha256(data).hexdigest():
        raise ValueError("metadata hash does not match the provided original bytes")
    edits = analyze_edit_candidates(signal, metadata["analysis_sample_rate_hz"])

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "case_id": None,  # assigned by the host application
        "module": "audio",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": metadata,
        "analysis": {
            "signal_edit_screening": edits,
            "synthetic_speech_screening": {
                "status": "not_run",
                "message": "Optional pretrained model was not requested.",
                "findings": [],
            },
        },
        "overall_interpretation": (
            "Triage findings only. This report does not establish authenticity or deliberate tampering. "
            "An investigator should examine the original evidence and corroborating information."
        ),
    }

    if include_synthetic_screening:
        synthetic = screen_synthetic_speech(
            signal,
            sample_rate=metadata["analysis_sample_rate_hz"],
            model_id=model_id,
        )
        result["analysis"]["synthetic_speech_screening"] = synthetic

    return result


def analyze_audio(
    data: bytes,
    filename: str,
    *,
    include_synthetic_screening: bool = False,
    model_id: str = DEFAULT_SYNTHETIC_MODEL_ID,
) -> dict[str, Any]:
    """Public integration entry point; validate, decode, then analyse one audio file."""
    signal, metadata = load_audio_bytes(data, filename)
    return analyze_decoded_audio(
        data, filename, signal, metadata,
        include_synthetic_screening=include_synthetic_screening, model_id=model_id,
    )
