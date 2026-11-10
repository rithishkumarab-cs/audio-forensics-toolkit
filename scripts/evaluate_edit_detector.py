"""Controlled audio-edit benchmark for the existing detector.

Reads a clean WAV, creates labelled test variants, runs the current
detector and writes CSV/JSON results. The source recording is not changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_forensics.edit_analysis import analyze_edit_candidates


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(path, np.clip(audio, -1, 1), sr, subtype="PCM_16")


def normalize_candidates(result: dict) -> list[dict]:
    output = []
    for f in result.get("findings", []):
        output.append({
            "start_s": float(f.get("timestamp_start_seconds", 0)),
            "end_s": float(f.get("timestamp_end_seconds", 0)),
            "peak_s": float(f.get("peak_timestamp_seconds", 0)),
            "type": f.get("finding_type", "unknown"),
            "priority": f.get("review_priority", "unknown"),
            "score": f.get("score"),
        })
    return output


def evaluate_boundaries(
    expected: list[float],
    candidates: list[dict],
    tolerance: float,
) -> tuple[list[float], list[float], list[dict]]:
    """Match each known boundary to at most one candidate interval."""
    unused = set(range(len(candidates)))
    matched = []
    missed = []

    for boundary in expected:
        possible = []
        for idx in unused:
            c = candidates[idx]
            if c["start_s"] - tolerance <= boundary <= c["end_s"] + tolerance:
                distance = abs(c["peak_s"] - boundary)
                possible.append((distance, idx))

        if not possible:
            missed.append(round(boundary, 3))
            continue

        distance, chosen = min(possible)
        unused.remove(chosen)
        matched.append({
            "expected_boundary_s": round(boundary, 3),
            "detected_peak_s": round(candidates[chosen]["peak_s"], 3),
            "peak_error_s": round(distance, 3),
        })

    extras = [candidates[i] for i in sorted(unused)]
    return missed, [c["peak_s"] for c in extras], matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a clean WAV recording")
    parser.add_argument(
        "--output-dir",
        default="reports/real_audio_test",
        help="Directory for test recordings and reports",
    )
    parser.add_argument("--edit-time", type=float, default=10.0)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--tolerance", type=float, default=0.35)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Input file not found: {source}")

    raw_bytes = source.read_bytes()
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1, dtype=np.float32)

    duration = len(audio) / sr
    if duration < 5:
        parser.error("Use a clean WAV recording of at least 5 seconds.")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Keep the edit safely inside the recording.
    segment_seconds = min(args.segment_seconds, duration * 0.30)
    segment_len = max(1, int(segment_seconds * sr))
    edit_time = float(np.clip(args.edit_time, 1.0, duration - segment_seconds - 1.0))
    edit_idx = int(edit_time * sr)
    edit_idx = min(edit_idx, len(audio) - segment_len - sr)
    edit_time = edit_idx / sr
    segment_seconds = segment_len / sr

    # Choose a distant audio region for replacement and hard-splice tests.
    source_idx = int((duration * (0.25 if edit_time > duration * 0.5 else 0.70)) * sr)
    source_idx = min(max(0, source_idx), len(audio) - segment_len - 1)
    if abs(source_idx - edit_idx) < segment_len:
        source_idx = max(0, edit_idx - 2 * segment_len)

    insert_idx = min(int(duration * 0.65 * sr), len(audio) - 1)
    duplicate_len = min(int(0.5 * sr), max(1, len(audio) // 20))
    duplicate_start = max(0, int(duration * 0.20 * sr))
    duplicate = audio[duplicate_start:duplicate_start + duplicate_len]

    rng = np.random.default_rng(20261025)
    noise_len = min(int(1.0 * sr), len(audio) // 5)
    noise_start = min(edit_idx, len(audio) - noise_len)
    noise = rng.normal(
        0, max(0.0005, float(np.sqrt(np.mean(audio ** 2))) * 0.05),
        noise_len,
    ).astype(np.float32)
    fade = min(int(0.1 * sr), max(1, noise_len // 4))
    if fade > 1:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        noise[:fade] *= ramp
        noise[-fade:] *= ramp[::-1]
    noise_control = audio.copy()
    noise_control[noise_start:noise_start + noise_len] += noise

    cases = [
        ("clean_baseline", audio.copy(), [], "control"),
        (
            "cut_join",
            np.concatenate((audio[:edit_idx], audio[edit_idx + segment_len:])),
            [edit_time],
            "edited",
        ),
        (
            "hard_splice",
            np.concatenate((audio[:edit_idx], audio[source_idx:])),
            [edit_time],
            "edited",
        ),
        (
            "replaced_segment",
            np.concatenate((
                audio[:edit_idx],
                audio[source_idx:source_idx + segment_len],
                audio[edit_idx + segment_len:],
            )),
            [edit_time, edit_time + segment_seconds],
            "edited",
        ),
        (
            "inserted_silence",
            np.concatenate((
                audio[:edit_idx],
                np.zeros(segment_len, dtype=np.float32),
                audio[edit_idx:],
            )),
            [edit_time, edit_time + segment_seconds],
            "edited",
        ),
        (
            "duplicated_segment",
            np.concatenate((
                audio[:insert_idx], duplicate, audio[insert_idx:],
            )),
            [insert_idx / sr, (insert_idx + len(duplicate)) / sr],
            "edited",
        ),
        ("noise_control", noise_control, [], "control"),
    ]

    summary_rows = []
    details = []

    for name, variant, expected, kind in cases:
        wav_path = out / f"{name}.wav"
        write_wav(wav_path, variant, sr)

        result = analyze_edit_candidates(variant, sr)
        candidates = normalize_candidates(result)
        missed, extra_peaks, matches = evaluate_boundaries(
            expected, candidates, args.tolerance
        )

        row = {
            "case": name,
            "kind": kind,
            "expected_boundaries_s": json.dumps(
                [round(x, 3) for x in expected]
            ),
            "matched_boundaries": len(matches),
            "missed_boundaries_s": json.dumps(missed),
            "candidate_count": len(candidates),
            "extra_candidate_peaks_s": json.dumps(
                [round(x, 3) for x in extra_peaks]
            ),
        }
        summary_rows.append(row)
        details.append({
            **row,
            "file": str(wav_path),
            "matches": matches,
            "findings": candidates,
            "detector_status": result.get("status"),
            "detector_message": result.get("message"),
        })

    csv_path = out / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = out / "benchmark_details.json"
    json_path.write_text(json.dumps({
        "source_file": source.name,
        "source_sha256": source_hash,
        "source_duration_seconds": round(duration, 3),
        "sample_rate_hz": sr,
        "boundary_tolerance_seconds": args.tolerance,
        "warning": (
            "This small controlled benchmark is diagnostic, not proof "
            "of forensic accuracy. Smooth edits may be undetectable "
            "from signal discontinuities alone."
        ),
        "cases": details,
    }, indent=2), encoding="utf-8")

    print(f"Source: {source.name} ({duration:.1f}s)")
    print(f"Source SHA-256: {source_hash}")
    print(f"CSV report: {csv_path}")
    print(f"Detailed JSON: {json_path}")
    print()
    for row in summary_rows:
        print(
            f"{row['case']:20} "
            f"matched={row['matched_boundaries']} "
            f"missed={row['missed_boundaries_s']} "
            f"candidates={row['candidate_count']} "
            f"extra={row['extra_candidate_peaks_s']}"
        )
    print("\nOriginal file was read only; all variants were written separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())