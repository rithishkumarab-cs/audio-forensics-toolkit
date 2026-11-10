from __future__ import annotations

import numpy as np

from audio_forensics.edit_analysis import analyze_edit_candidates


def speech_like_signal(sr: int = 16000, duration: int = 16) -> np.ndarray:
    """Create deterministic voiced audio for repeatable regression tests."""
    rng = np.random.default_rng(2026)
    count = sr * duration
    t = np.arange(count, dtype=np.float64) / sr

    pitch = 125 + 28 * np.sin(2 * np.pi * 0.7 * t)
    pitch += 10 * np.sin(2 * np.pi * 2.1 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / sr

    voiced = np.zeros(count, dtype=np.float64)
    for harmonic, amplitude in (
        (1, 1.0), (2, 0.45), (3, 0.28),
        (4, 0.18), (5, 0.10), (6, 0.08),
    ):
        voiced += amplitude * np.sin(harmonic * phase + 0.2 * harmonic)

    syllable = t % 0.42
    envelope = 0.015 + 0.15 * np.sin(np.pi * syllable / 0.42) ** 1.2
    envelope *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.17 * t)

    x = voiced * envelope + rng.normal(0, 0.0015, count)

    for time_s in (1.3, 3.2, 5.7, 9.1, 12.4, 14.2):
        begin = int(time_s * sr)
        end = min(count, begin + int(0.07 * sr))
        x[begin:end] += rng.normal(0, 0.035, end - begin)

    x = x / max(np.max(np.abs(x)), 1e-8) * 0.8
    return x.astype(np.float32)


def nearest_peak_error(findings: list[dict], boundary: float) -> float:
    if not findings:
        return float("inf")
    return min(
        abs(float(f["peak_timestamp_seconds"]) - boundary)
        for f in findings
    )


def match_boundaries(expected, findings, tolerance=0.10):
    """Match known boundaries to candidate PEAKS, not padded intervals."""
    remaining = list(findings)
    matches, missed = [], []

    for boundary in expected:
        options = [
            (
                abs(float(f["peak_timestamp_seconds"]) - boundary),
                i,
                f,
            )
            for i, f in enumerate(remaining)
        ]
        options = [x for x in options if x[0] <= tolerance]

        if not options:
            missed.append(boundary)
            continue

        error, index, finding = min(options, key=lambda x: x[0])
        matches.append({
            "expected": boundary,
            "detected": finding["peak_timestamp_seconds"],
            "error": error,
        })
        remaining.pop(index)

    return matches, missed, remaining


def test_clean_speech_like_reference_does_not_flag_every_syllable():
    result = analyze_edit_candidates(speech_like_signal(), 16000)
    assert result["status"] == "completed"
    assert len(result["findings"]) <= 5, [
        f["peak_timestamp_seconds"] for f in result["findings"]
    ]


def test_segment_replacement_checks_both_known_boundaries():
    sr = 16000
    clean = speech_like_signal(sr)
    start, source, length = 6 * sr, 11 * sr, 2 * sr

    edited = np.concatenate((
        clean[:start],
        clean[source:source + length],
        clean[start + length:],
    )).astype(np.float32)

    findings = analyze_edit_candidates(edited, sr)["findings"]

    assert nearest_peak_error(findings, 6.0) <= 0.10, [
        f["peak_timestamp_seconds"] for f in findings
    ]
    assert nearest_peak_error(findings, 8.0) <= 0.10, [
        f["peak_timestamp_seconds"] for f in findings
    ]


def test_peak_far_from_boundary_is_not_counted_as_detection():
    findings = [
        {
            "timestamp_start_seconds": 7.95,
            "timestamp_end_seconds": 8.30,
            "peak_timestamp_seconds": 8.20,
        }
    ]
    matches, missed, extras = match_boundaries(
        [7.80], findings, tolerance=0.10
    )
    assert matches == []
    assert missed == [7.80]
    assert len(extras) == 1


def test_peak_within_tolerance_is_matched_and_error_reported():
    findings = [
        {
            "timestamp_start_seconds": 5.96,
            "timestamp_end_seconds": 6.04,
            "peak_timestamp_seconds": 6.003,
        }
    ]
    matches, missed, extras = match_boundaries(
        [6.0], findings, tolerance=0.10
    )
    assert len(matches) == 1
    assert not missed
    assert not extras
    assert matches[0]["error"] <= 0.01


def test_candidate_interval_tracks_peak_without_arbitrary_wide_padding():
    sr = 16000
    rng = np.random.default_rng(55)
    clean = (0.02 * rng.normal(size=sr * 8)).astype(np.float32)
    boundary = 4 * sr
    edited = np.concatenate((
        clean[:boundary],
        clean[boundary + int(0.75 * sr):],
    )).astype(np.float32)

    result = analyze_edit_candidates(edited, sr)
    near = [
        f for f in result["findings"]
        if abs(f["peak_timestamp_seconds"] - 4.0) <= 0.10
    ]
    assert near, [f["peak_timestamp_seconds"] for f in result["findings"]]
    assert max(
        f["timestamp_end_seconds"] - f["timestamp_start_seconds"]
        for f in near
    ) <= 0.12


def test_smooth_pitch_glide_does_not_create_a_signal_discontinuity_finding():
    sr = 16000
    t = np.arange(sr * 6, dtype=np.float32) / sr
    frequency = np.full_like(t, 220.0)

    mask = (t >= 2.0) & (t <= 4.0)
    progress = (t[mask] - 2.0) / 2.0
    frequency[mask] = 220 + 100 * progress**2 * (3 - 2 * progress)
    frequency[t > 4.0] = 320.0

    phase = 2 * np.pi * np.cumsum(frequency, dtype=np.float64) / sr
    signal = (0.15 * np.sin(phase)).astype(np.float32)

    result = analyze_edit_candidates(signal, sr)
    assert not any(
        f["finding_type"] == "candidate_signal_discontinuity"
        for f in result["findings"]
    )


def test_speech_replacement_findings_report_the_new_spectral_cue():
    sr = 16000
    clean = speech_like_signal(sr)
    start, source, length = 6 * sr, 11 * sr, 2 * sr

    edited = np.concatenate((
        clean[:start],
        clean[source:source + length],
        clean[start + length:],
    )).astype(np.float32)

    findings = analyze_edit_candidates(edited, sr)["findings"]

    for boundary in (6.0, 8.0):
        assert nearest_peak_error(findings, boundary) <= 0.10
        nearest = min(
            findings,
            key=lambda f: abs(f["peak_timestamp_seconds"] - boundary),
        )
        cues = nearest["supporting_features"]["cues"]
        assert "pre/post spectral-context discontinuity" in cues
        assert nearest["score_type"].endswith("not_probability")


def test_pcm16_quantisation_preserves_known_replacement_boundaries():
    sr = 16000
    clean = speech_like_signal(sr)
    clean = (
        np.round(np.clip(clean, -1.0, 1.0) * 32767.0) / 32768.0
    ).astype(np.float32)

    start, source, length = 6 * sr, 11 * sr, 2 * sr
    edited = np.concatenate((
        clean[:start],
        clean[source:source + length],
        clean[start + length:],
    )).astype(np.float32)

    findings = analyze_edit_candidates(edited, sr)["findings"]
    assert nearest_peak_error(findings, 6.0) <= 0.10
    assert nearest_peak_error(findings, 8.0) <= 0.10


def test_low_level_noise_does_not_shift_replacement_boundaries():
    sr = 16000
    clean = speech_like_signal(sr)
    start, source, length = 6 * sr, 11 * sr, 2 * sr

    edited = np.concatenate((
        clean[:start],
        clean[source:source + length],
        clean[start + length:],
    )).astype(np.float32)

    rng = np.random.default_rng(77)
    edited += rng.normal(0, 0.001, edited.size).astype(np.float32)

    findings = analyze_edit_candidates(edited, sr)["findings"]
    assert nearest_peak_error(findings, 6.0) <= 0.10
    assert nearest_peak_error(findings, 8.0) <= 0.10


def test_detection_results_are_deterministic_for_identical_audio():
    signal = speech_like_signal()
    first = analyze_edit_candidates(signal, 16000)
    second = analyze_edit_candidates(signal, 16000)

    compact = lambda result: [
        (
            f["finding_type"],
            f["timestamp_start_seconds"],
            f["timestamp_end_seconds"],
            f["peak_timestamp_seconds"],
            f["score"],
            f["supporting_cue_count"],
        )
        for f in result["findings"]
    ]

    assert compact(first) == compact(second)


def test_reported_scores_do_not_exceed_robust_score_cap():
    result = analyze_edit_candidates(speech_like_signal(), 16000)

    for finding in result["findings"]:
        score = finding["score"]
        assert score is None or abs(float(score)) <= 50.0