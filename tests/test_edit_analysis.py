import numpy as np
from audio_forensics.edit_analysis import analyze_edit_candidates


def test_short_signal_is_inconclusive():
    result = analyze_edit_candidates(np.zeros(1000, dtype=np.float32), 16000)
    assert result["status"] == "inconclusive"
    assert result["findings"] == []


def test_silence_is_inconclusive_not_fake_or_edited():
    result = analyze_edit_candidates(np.zeros(16000, dtype=np.float32), 16000)
    assert result["status"] == "inconclusive"
    assert result["findings"] == []
    assert "silent" in result["message"].lower()


def test_smooth_tone_has_no_candidate_intervals():
    sr = 16000
    t = np.arange(sr * 4, dtype=np.float32) / sr
    # Smooth amplitude modulation with no inserted discontinuity.
    envelope = 0.25 + 0.15 * np.sin(2 * np.pi * 0.4 * t)
    x = envelope * np.sin(2 * np.pi * 220 * t)
    result = analyze_edit_candidates(x, sr)
    assert result["status"] == "completed"
    assert result["findings"] == []


def test_controlled_inserted_silence_is_flagged_for_review():
    sr = 16000
    rng = np.random.default_rng(7)
    x = (0.02 * rng.normal(size=sr * 4)).astype(np.float32)
    # Controlled edit-like discontinuity: a 250 ms silent segment in a noisy recording.
    x[sr * 2: sr * 2 + int(0.25 * sr)] = 0.0
    result = analyze_edit_candidates(x, sr)
    assert result["status"] == "completed"
    assert result["findings"], "Expected controlled abrupt transition to be flagged for review"
    assert all("not_probability" in f["score_type"] for f in result["findings"])
    assert all(f["supporting_cue_count"] >= 2 for f in result["findings"])
    assert any(1.5 <= f["peak_timestamp_seconds"] <= 2.8 for f in result["findings"])


def test_output_states_heuristic_score_is_not_probability():
    sr = 16000
    rng = np.random.default_rng(8)
    x = (0.01 * rng.normal(size=sr * 3)).astype(np.float32)
    x[sr:sr + 3200] = 0.0
    result = analyze_edit_candidates(x, sr)
    assert result["method"] == "multi_cue_signal_discontinuity_screening"
    assert "not prove" in " ".join(result["limitations"]).lower()
    for finding in result["findings"]:
        assert "not_probability" in finding["score_type"] or "not_probability" in finding["score_type"]
        assert 0 <= finding["timestamp_start_seconds"] <= 3


def test_separate_silence_boundaries_are_not_merged_into_one_event():
    sr = 16000
    rng = np.random.default_rng(27)
    x = (0.015 * rng.normal(size=sr * 5)).astype(np.float32)
    start = int(2.0 * sr)
    width = int(0.45 * sr)
    x[start:start + width] = 0.0

    result = analyze_edit_candidates(x, sr)
    peaks = [f["peak_timestamp_seconds"] for f in result["findings"]]

    assert any(abs(t - 2.0) <= 0.20 for t in peaks), (
        f"Missing silence-entry boundary: {peaks}"
    )
    assert any(abs(t - 2.45) <= 0.20 for t in peaks), (
        f"Missing silence-exit boundary: {peaks}"
    )


def test_hard_splice_between_different_audio_segments_is_localised():
    sr = 16000
    rng = np.random.default_rng(42)
    duration = 8.0
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    x = np.zeros_like(t)

    segments = [
        (0.0, 2.0, 150.0, 0.20),
        (2.0, 4.0, 210.0, 0.15),
        (4.0, 6.0, 300.0, 0.18),
        (6.0, 8.0, 500.0, 0.12),
    ]

    for start, end, freq, amp in segments:
        mask = (t >= start) & (t < end)
        local_t = t[mask] - start
        x[mask] = amp * (
            np.sin(2 * np.pi * freq * local_t)
            + 0.2 * np.sin(2 * np.pi * 3 * freq * local_t)
        )

    x += (rng.normal(size=x.size) * 0.002).astype(np.float32)

    splice_time = 1.5
    source_time = 6.54
    splice_idx = int(splice_time * sr)
    source_idx = int(source_time * sr)

    edited = np.concatenate(
        (x[:splice_idx], x[source_idx:])
    ).astype(np.float32)

    result = analyze_edit_candidates(edited, sr)
    peaks = [f["peak_timestamp_seconds"] for f in result["findings"]]

    assert any(abs(t - splice_time) <= 0.20 for t in peaks), (
        f"Expected a candidate near the known splice at "
        f"{splice_time}s; candidates={peaks}"
    )


def test_non_finite_audio_returns_inconclusive_without_crashing():
    sr = 16000
    x = np.full(sr, 0.1, dtype=np.float32)
    x[sr // 2] = np.nan

    result = analyze_edit_candidates(x, sr)

    assert result["status"] == "inconclusive"
    assert not result["findings"]


def test_near_silence_transition_is_labelled_as_context_not_confirmed_edit():
    sr = 16000
    rng = np.random.default_rng(29)
    x = (0.02 * rng.normal(size=sr * 4)).astype(np.float32)
    x[sr * 2:sr * 2 + int(0.4 * sr)] = 0.0

    result = analyze_edit_candidates(x, sr)

    assert any(
        f["finding_type"] == "candidate_silence_transition"
        for f in result["findings"]
    )

    for finding in result["findings"]:
        if finding["finding_type"] == "candidate_silence_transition":
            assert finding["review_priority"] == "context"
            assert "normal pause" in finding["explanation"].lower()


def test_smooth_pitch_glide_is_not_flagged_as_an_edit_boundary():
    sr = 16000
    duration = 6.0
    t = np.arange(int(sr * duration), dtype=np.float32) / sr

    start, end = 2.0, 4.0
    freq = np.full_like(t, 220.0)
    mask = (t >= start) & (t <= end)
    progress = (t[mask] - start) / (end - start)

    freq[mask] = 220.0 + 100.0 * (
        progress * progress * (3.0 - 2.0 * progress)
    )
    freq[t > end] = 320.0

    phase = 2 * np.pi * np.cumsum(freq, dtype=np.float64) / sr
    x = (0.15 * np.sin(phase)).astype(np.float32)

    result = analyze_edit_candidates(x, sr)

    edit_like = [
        f for f in result["findings"]
        if f["finding_type"] == "candidate_signal_discontinuity"
    ]

    assert not edit_like, (
        f"Smooth pitch change should not be tagged as an edit: {edit_like}"
    )


def test_speech_to_low_level_background_noise_reports_both_context_boundaries():
    sr = 16000
    rng = np.random.default_rng(101)
    t = np.arange(sr * 6, dtype=np.float32) / sr

    active_audio = (
        0.15 * np.sin(2 * np.pi * 220 * t)
        + 0.01 * rng.normal(size=t.size)
    ).astype(np.float32)

    ambience = (0.003 * rng.normal(size=t.size)).astype(np.float32)
    x = active_audio.copy()

    start, end = 2.0, 4.0
    x[int(start * sr):int(end * sr)] = (
        ambience[int(start * sr):int(end * sr)]
    )

    result = analyze_edit_candidates(x, sr)

    events = [
        f for f in result["findings"]
        if f["finding_type"] == "candidate_activity_transition"
    ]
    peaks = [f["peak_timestamp_seconds"] for f in events]

    assert any(abs(p - start) <= 0.20 for p in peaks), (
        f"Missing active-to-ambience boundary: {peaks}"
    )
    assert any(abs(p - end) <= 0.20 for p in peaks), (
        f"Missing ambience-to-active boundary: {peaks}"
    )
    assert all(f["review_priority"] == "context" for f in events)
