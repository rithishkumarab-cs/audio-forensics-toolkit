"""Audio edit-boundary screening using multiple signal cues.

Experimental forensic triage only. Findings require human review and
must not be interpreted as proof that audio was manipulated.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft

METHOD = "multi_cue_signal_discontinuity_screening"


def _robust_z(values: np.ndarray, scale_floor: float = 1e-8) -> np.ndarray:
    """Robust z-like score using median absolute deviation."""
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, scale_floor)
    return (values - median) / scale


def _merge_candidate_indices(
    indices: np.ndarray, min_gap_frames: int
) -> list[list[int]]:
    """Group nearby candidate frames without merging distant boundaries."""
    if indices.size == 0:
        return []

    groups: list[list[int]] = [[int(indices[0])]]

    for raw_index in indices[1:]:
        index = int(raw_index)

        if index - groups[-1][-1] <= min_gap_frames:
            groups[-1].append(index)
        else:
            groups.append([index])

    return groups


def _pre_post_spectral_distance(
    magnitude: np.ndarray,
    spec_times: np.ndarray,
    *,
    side_window_seconds: float = 0.06,
    exclusion_gap_seconds: float = 0.02,
) -> np.ndarray:
    """Compare normalised spectral shapes on either side of a candidate join."""
    magnitude = np.asarray(magnitude, dtype=np.float64)
    times = np.asarray(spec_times, dtype=np.float64)

    if (
        magnitude.ndim != 2
        or magnitude.shape[1] != times.size
        or times.size < 8
    ):
        return np.zeros(times.size, dtype=np.float64)

    step = float(np.median(np.diff(times))) if times.size > 1 else 0.0

    if not np.isfinite(step) or step <= 0:
        return np.zeros(times.size, dtype=np.float64)

    side_frames = max(3, int(round(side_window_seconds / step)))
    gap_frames = max(1, int(round(exclusion_gap_seconds / step)))
    flank = side_frames + gap_frames

    distances = np.zeros(times.size, dtype=np.float64)

    for index in range(flank, times.size - flank):
        left = np.mean(
            magnitude[
                :,
                index - gap_frames - side_frames:index - gap_frames,
            ],
            axis=1,
        )
        right = np.mean(
            magnitude[
                :,
                index + gap_frames:index + gap_frames + side_frames,
            ],
            axis=1,
        )

        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))

        if denominator > 1e-12:
            cosine = float(np.dot(left, right) / denominator)
            distances[index] = float(np.clip(1.0 - cosine, 0.0, 1.0))

    return distances


def _empty_result(
    status: str,
    message: str,
    limitations: list[str],
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "status": status,
        "message": message,
        "parameters": {},
        "features": {},
        "findings": [],
        "candidate_events_suppressed": 0,
        "limitations": limitations,
    }


def analyze_edit_candidates(
    signal: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = 25.0,
    hop_ms: float = 10.0,
    z_threshold: float = 7.0,
    merge_gap_seconds: float = 0.08,
    edge_guard_seconds: float = 0.25,
) -> dict[str, Any]:
    """Find localised signal-change candidates for human review.

    Cues include RMS changes, spectral flux, sample jumps, and spectral
    differences before/after a possible join.

    The score is a robust anomaly score, not a probability of editing.
    """

    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    duration = float(x.size / sample_rate) if sample_rate > 0 else 0.0

    limitations = [
        "This is experimental signal-level triage, not a validated general-purpose audio-splice detector.",
        "Natural speech transients, pauses, humming, microphone changes, noise reduction, and codec artifacts can trigger candidates.",
        "Smooth edits may not produce a detectable discontinuity; absence of findings does not prove the recording is unedited.",
        "This screen cannot reconstruct missing words or establish what was removed without corroborating evidence or a trusted reference.",
    ]

    if sample_rate <= 0 or x.size < int(max(sample_rate, 1) * 0.5):
        return _empty_result(
            "inconclusive",
            "Audio is too short for stable signal-change screening.",
            limitations,
        )

    if not np.all(np.isfinite(x)):
        return _empty_result(
            "inconclusive",
            "Audio contains non-finite samples and cannot be analysed reliably.",
            limitations,
        )

    if float(np.max(np.abs(x))) < 1e-7:
        return _empty_result(
            "inconclusive",
            "Audio is silent or too close to silence for edit screening.",
            limitations,
        )

    frame_len = max(64, int(round(sample_rate * window_ms / 1000.0)))
    hop = max(16, int(round(sample_rate * hop_ms / 1000.0)))

    if x.size < frame_len * 2:
        return _empty_result(
            "inconclusive",
            "Audio is too short for stable screening.",
            limitations,
        )

    starts = np.arange(0, x.size - frame_len + 1, hop, dtype=np.int64)

    frames = np.lib.stride_tricks.sliding_window_view(
        x, frame_len
    )[::hop][:len(starts)]

    rms = np.sqrt(
        np.mean(np.square(frames, dtype=np.float32), axis=1) + 1e-12
    )
    rms_db = 20.0 * np.log10(rms + 1e-8)
    rms_delta = np.abs(np.diff(rms_db, prepend=rms_db[0]))

    center_times = starts / sample_rate + frame_len / (2.0 * sample_rate)

    nperseg = min(512, max(128, frame_len))
    noverlap = min(nperseg - 1, int(nperseg * 0.75))

    _, spec_times, spectrum = stft(
        x,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )

    magnitude = np.abs(spectrum).astype(np.float32)
    magnitude /= np.maximum(
        np.sum(magnitude, axis=0, keepdims=True),
        1e-8,
    )

    flux = np.mean(
        np.maximum(
            np.diff(
                magnitude,
                axis=1,
                prepend=magnitude[:, :1],
            ),
            0.0,
        ),
        axis=0,
    )

    flux_at_frames = np.interp(
        center_times,
        spec_times,
        flux,
        left=float(flux[0]),
        right=float(flux[-1]),
    )

    # Sample-level discontinuity relative to local RMS.
    abs_diff = np.abs(
        np.diff(x.astype(np.float64), prepend=float(x[0]))
    )
    half_hop = max(1, hop // 2)
    jump_ratio = np.zeros(len(starts), dtype=np.float64)
    jump_amplitude = np.zeros(len(starts), dtype=np.float64)

    for frame_index, center in enumerate(starts + frame_len // 2):
        lo = max(1, int(center) - half_hop)
        hi = min(len(abs_diff), int(center) + half_hop + 1)

        local_jump = (
            float(np.max(abs_diff[lo:hi])
            if hi > lo else 0.0)
        )

        jump_amplitude[frame_index] = local_jump
        jump_ratio[frame_index] = (
            local_jump / (float(rms[frame_index]) + 1e-6)
        )

    def robust_scores(
        values: np.ndarray,
        local_window_frames: int = 151,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate local and global robust scores with bounded values."""
        values = np.asarray(values, dtype=np.float64)

        global_median = float(np.median(values))
        global_scale = max(
            1.4826
            * float(np.median(np.abs(values - global_median))),
            1e-8,
        )

        global_score = np.clip(_robust_z(values), -50.0, 50.0)

        size = min(
            local_window_frames,
            len(values) if len(values) % 2 else max(1, len(values) - 1),
        )

        if size % 2 == 0:
            size = max(1, size - 1)

        if size < 3:
            return global_score, global_score

        local_median = median_filter(
            values, size=size, mode="nearest"
        )
        local_mad = median_filter(
            np.abs(values - local_median),
            size=size,
            mode="nearest",
        )

        local_scale = np.maximum(
            1.4826 * local_mad,
            max(global_scale * 0.25, 1e-8),
        )

        local_score = np.clip(
            (values - local_median) / local_scale,
            -50.0,
            50.0,
        )

        return local_score, global_score

    rms_z, _ = robust_scores(rms_delta)
    flux_z, _ = robust_scores(flux_at_frames)
    jump_local_z, jump_global_z = robust_scores(jump_ratio)
    jump_z = np.maximum(jump_local_z, jump_global_z)

    # Compare spectra from the left and right of each candidate boundary.
    context_distance_raw = _pre_post_spectral_distance(
        magnitude, spec_times
    )

    context_distance = np.interp(
        center_times,
        spec_times,
        context_distance_raw,
        left=float(context_distance_raw[0]),
        right=float(context_distance_raw[-1]),
    )

    context_z, _ = robust_scores(context_distance)

    context_z_threshold = max(z_threshold, 7.0)
    context_corroboration_threshold = max(2.5, z_threshold * 0.40)
    context_absolute_distance_threshold = 0.10

    # A fixed distance floor avoids extreme z-scores on steady tones.
    context_cue = (
        (context_z >= context_z_threshold)
        & (context_distance >= context_absolute_distance_threshold)
    )

    # Contextual active-audio ↔ silence/room-tone transitions.
    # These are review events, not declarations of tampering.
    context_frames = max(
        3, int(round(0.08 / (hop / sample_rate)))
    )

    rms_squared = np.square(rms.astype(np.float64))
    prefix_energy = np.concatenate(([0.0], np.cumsum(rms_squared)))
    frame_index = np.arange(len(rms), dtype=np.int64)

    left_lo = np.maximum(0, frame_index - context_frames - 1)
    left_hi = np.maximum(0, frame_index - 1)
    right_lo = np.minimum(len(rms), frame_index + 2)
    right_hi = np.minimum(
        len(rms), frame_index + context_frames + 1
    )

    left_count = np.maximum(1, left_hi - left_lo)
    right_count = np.maximum(1, right_hi - right_lo)

    left_rms = np.sqrt(
        np.maximum(
            (prefix_energy[left_hi] - prefix_energy[left_lo]) / left_count,
            0.0,
        )
    )
    right_rms = np.sqrt(
        np.maximum(
            (prefix_energy[right_hi] - prefix_energy[right_lo]) / right_count,
            0.0,
        )
    )

    level_step_db = np.abs(
        20.0 * np.log10(left_rms + 1e-8)
        - 20.0 * np.log10(right_rms + 1e-8)
    )

    overall_rms = float(
        np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12)
    )
    active_floor = max(0.002, overall_rms * 0.08)
    silence_floor = max(1e-4, overall_rms * 0.01)

    near_silence = (
        (level_step_db >= 18.0)
        & (np.minimum(left_rms, right_rms) < silence_floor)
        & (np.maximum(left_rms, right_rms) > active_floor)
    )

    activity_transition = (
        (level_step_db >= 14.0)
        & (
            np.minimum(left_rms, right_rms)
            < max(1e-4, overall_rms * 0.20)
        )
        & (np.maximum(left_rms, right_rms) > active_floor)
        & ~near_silence
    )

    contextual_level_transition = near_silence | activity_transition

    rms_cue = rms_z >= z_threshold
    flux_cue = flux_z >= z_threshold
    raw_jump_cue = jump_ratio >= 2.0
    jump_cue = (jump_z >= z_threshold) | raw_jump_cue

    cue_matrix = np.vstack((
        rms_cue,
        flux_cue,
        jump_cue,
        context_cue,
    ))
    cue_count = np.sum(cue_matrix, axis=0)

    moderate_threshold = max(3.0, z_threshold * 0.60)
    raw_jump_corroboration_threshold = max(2.5, z_threshold * 0.45)

    # A very large sample jump is itself a candidate boundary.
    # Keep this rule narrow to avoid lowering every detector threshold.
    ratio_edge_candidate = (
        (jump_ratio >= 2.0)
        & (
            (rms_z >= moderate_threshold)
            | (flux_z >= raw_jump_corroboration_threshold)
            | (jump_ratio >= 8.0)
        )
    )

    # Catch large absolute jumps when RMS change also supports a boundary.
    absolute_jump_candidate = jump_amplitude >= 0.65

    raw_edge_candidate = (
        ratio_edge_candidate | absolute_jump_candidate
    )

    strong_edge_candidate = (
        jump_cue
        & (
            (rms_z >= moderate_threshold)
            | (flux_z >= moderate_threshold)
        )
    ) | raw_edge_candidate

    context_spectral_candidate = context_cue & (
        (rms_z >= context_corroboration_threshold)
        | (flux_z >= context_corroboration_threshold)
        | (jump_z >= context_corroboration_threshold)
    )

    low_activity = rms < active_floor

    generic_hard_fallback = (
        ((rms_z >= 12.0) & (jump_z >= 4.0) & ~low_activity)
        | (low_activity & (jump_ratio >= 8.0))
    )

    hard_boundary_fallback = (
        generic_hard_fallback | contextual_level_transition
    )

    effective_cue_count = np.maximum(
        cue_count, raw_edge_candidate.astype(np.int8) * 2
    )
    effective_cue_count = np.maximum(
        effective_cue_count, strong_edge_candidate.astype(np.int8) * 2
    )
    effective_cue_count = np.maximum(
        effective_cue_count, hard_boundary_fallback.astype(np.int8) * 2
    )
    effective_cue_count = np.maximum(
        effective_cue_count,
        context_spectral_candidate.astype(np.int8) * 2,
    )

    valid = (
        (center_times >= edge_guard_seconds)
        & (center_times <= duration - edge_guard_seconds)
    )

    strong_edge_in_low_activity = (
        strong_edge_candidate & (jump_ratio >= 8.0)
    )

    signal_candidate = (
        (
            strong_edge_candidate
            & (~low_activity | strong_edge_in_low_activity)
        )
        | ((cue_count >= 3) & ~low_activity)
        | (context_spectral_candidate & ~low_activity)
    )

    candidate_mask = (
        signal_candidate | hard_boundary_fallback
    ) & valid

    candidate_indices = np.flatnonzero(candidate_mask)

    min_gap_frames = max(
        1, int(round(merge_gap_seconds / (hop / sample_rate)))
    )

    groups = _merge_candidate_indices(
        candidate_indices, min_gap_frames
    )

    findings: list[dict[str, Any]] = []

    time_resolution_pad = (
        frame_len / (2.0 * sample_rate)
        + hop / (2.0 * sample_rate)
    )

    for group in groups:
        def supported_scores(index: int) -> list[float]:
            scores: list[float] = []
            edge_corroborated = bool(jump_cue[index])

            if (
                rms_z[index] >= z_threshold
                or (
                    edge_corroborated
                    and rms_z[index] >= moderate_threshold
                )
                or (
                    context_spectral_candidate[index]
                    and rms_z[index] >= context_corroboration_threshold
                )
                or (
                    hard_boundary_fallback[index]
                    and rms_z[index] >= 12.0
                )
            ):
                scores.append(float(rms_z[index]))

            if (
                flux_z[index] >= z_threshold
                or (
                    edge_corroborated
                    and flux_z[index]
                    >= raw_jump_corroboration_threshold
                )
                or (
                    context_spectral_candidate[index]
                    and flux_z[index] >= context_corroboration_threshold
                )
            ):
                scores.append(float(flux_z[index]))

            if (
                jump_z[index] >= z_threshold
                or (
                    context_spectral_candidate[index]
                    and jump_z[index] >= context_corroboration_threshold
                )
                or (
                    hard_boundary_fallback[index]
                    and jump_z[index] >= 4.0
                )
            ):
                scores.append(float(jump_z[index]))

            if context_cue[index]:
                scores.append(float(context_z[index]))

            return scores

        def strength(index: int) -> float:
            values = supported_scores(index)
            return min(values) if values else float("-inf")

        peak_index = max(
            group,
            key=lambda index: (
                int(effective_cue_count[index]),
                strength(index),
            ),
        )

        cues: list[str] = []
        scores: list[float] = []

        if (
            rms_z[peak_index] >= z_threshold
            or (
                jump_cue[peak_index]
                and rms_z[peak_index] >= moderate_threshold
            )
            or (
                context_spectral_candidate[peak_index]
                and rms_z[peak_index] >= context_corroboration_threshold
            )
            or (
                hard_boundary_fallback[peak_index]
                and rms_z[peak_index] >= 12.0
            )
        ):
            cues.append("short-time sound-level change")
            scores.append(float(rms_z[peak_index]))

        if (
            flux_z[peak_index] >= z_threshold
            or (
                jump_cue[peak_index]
                and flux_z[peak_index] >= moderate_threshold
            )
            or (
                context_spectral_candidate[peak_index]
                and flux_z[peak_index] >= context_corroboration_threshold
            )
            or (
                raw_edge_candidate[peak_index]
                and flux_z[peak_index] >= raw_jump_corroboration_threshold
            )
        ):
            cues.append("spectral-content change")
            scores.append(float(flux_z[peak_index]))

        if (
            jump_z[peak_index] >= z_threshold
            or (
                context_spectral_candidate[peak_index]
                and jump_z[peak_index] >= context_corroboration_threshold
            )
            or (
                hard_boundary_fallback[peak_index]
                and jump_z[peak_index] >= 4.0
            )
            or raw_edge_candidate[peak_index]
        ):
            cues.append("sharp sample-level transition")

            if (
                jump_z[peak_index] >= z_threshold
                or (
                    context_spectral_candidate[peak_index]
                    and jump_z[peak_index] >= context_corroboration_threshold
                )
                or (
                    hard_boundary_fallback[peak_index]
                    and jump_z[peak_index] >= 4.0
                )
            ):
                scores.append(float(jump_z[peak_index]))

        if context_cue[peak_index]:
            cues.append("pre/post spectral-context discontinuity")
            scores.append(float(context_z[peak_index]))

        peak_time = float(center_times[peak_index])

        start_time = max(
            0.0,
            min(float(center_times[i]) for i in group) - time_resolution_pad,
        )
        end_time = min(
            duration,
            max(float(center_times[i]) for i in group) + time_resolution_pad,
        )

        is_silence = bool(near_silence[peak_index])
        is_activity = bool(activity_transition[peak_index])

        if is_silence:
            finding_type = "candidate_silence_transition"
            explanation = (
                "A sharp transition between near-silence and active audio occurs here. "
                "It may be a normal pause, recording dropout, or edit; audio level alone "
                "cannot distinguish these possibilities. Review the source."
            )
        elif is_activity:
            finding_type = "candidate_activity_transition"
            explanation = (
                "A large sound-level change separates active audio from a much quieter, "
                "non-silent region. This may be a normal speech pause, room-tone/background-"
                "noise change, or edit. Treat it as a contextual review point, not proof of tampering."
            )
        else:
            finding_type = "candidate_signal_discontinuity"
            explanation = (
                "Multiple signal-change indicators agree near this interval: "
                + ", ".join(cues)
                + ". Review the surrounding audio; this does not by itself establish editing."
            )

        findings.append({
            "finding_id": f"AUD-EDIT-{len(findings) + 1:03d}",
            "module": "audio",
            "finding_type": finding_type,
            "timestamp_start_seconds": round(start_time, 3),
            "timestamp_end_seconds": round(end_time, 3),
            "peak_timestamp_seconds": round(peak_time, 3),
            "score": round(min(scores), 3) if scores else None,
            "score_type": "minimum_supporting_robust_z_score_not_probability",
            "review_priority": (
                "context"
                if (is_silence or is_activity)
                else ("higher" if cue_count[peak_index] >= 3 else "review")
            ),
            "supporting_cue_count": int(effective_cue_count[peak_index]),
            "context_features": {
                "left_context_rms": round(float(left_rms[peak_index]), 7),
                "right_context_rms": round(float(right_rms[peak_index]), 7),
                "left_right_level_step_db": round(float(level_step_db[peak_index]), 3),
                "near_silence_transition": is_silence,
                "activity_level_transition": is_activity,
            },
            "supporting_features": {
                "rms_change_robust_z": round(float(rms_z[peak_index]), 3),
                "spectral_flux_robust_z": round(float(flux_z[peak_index]), 3),
                "sample_jump_robust_z": round(float(jump_z[peak_index]), 3),
                "pre_post_spectral_context_distance": round(
                    float(context_distance[peak_index]), 5
                ),
                "pre_post_spectral_context_robust_z": round(
                    float(context_z[peak_index]), 3
                ),
                "cues": cues,
            },
            "explanation": explanation,
            "limitations": (
                "Speech transients, abrupt sounds, microphone changes and codec artefacts "
                "can also trigger signal-change indicators."
            ),
        })

    # Present higher-priority events first, then re-number consistently.
    priority_order = {"higher": 0, "review": 1, "context": 2}

    findings.sort(key=lambda item: (
        priority_order.get(item["review_priority"], 3),
        -float(item["score"]) if item["score"] is not None else float("inf"),
    ))

    for number, finding in enumerate(findings, start=1):
        finding["finding_id"] = f"AUD-EDIT-{number:03d}"

    suppressed_indices = np.flatnonzero(
        (
            (cue_count == 1)
            | (rms_cue & flux_cue & ~jump_cue)
        )
        & ~hard_boundary_fallback
        & valid
    )

    suppressed_groups = _merge_candidate_indices(
        suppressed_indices, min_gap_frames
    )

    return {
        "method": METHOD,
        "status": "completed",
        "message": (
            "No signal-change events crossed the screening thresholds. "
            "This does not prove the recording is unedited."
            if not findings
            else (
                f"{len(findings)} signal-change review event(s) were found. "
                "These are triage indicators, not confirmed edits."
            )
        ),
        "parameters": {
            "frame_window_ms": window_ms,
            "hop_ms": hop_ms,
            "robust_z_threshold": z_threshold,
            "candidate_rule": (
                "sample_jump_with_corroboration, three_or_more_of_four_cues, "
                "pre_post_spectral_context_with_corroboration, or contextual_level_transition"
            ),
            "pre_post_spectral_context": (
                "cosine distance between mean normalised spectra 60 ms before and "
                "after a 20 ms boundary exclusion gap; requires another moderately elevated cue"
            ),
            "pre_post_spectral_context_z_threshold": context_z_threshold,
            "pre_post_spectral_context_absolute_distance_threshold": (
                context_absolute_distance_threshold
            ),
            "pre_post_spectral_context_corroboration_threshold": (
                context_corroboration_threshold
            ),
            "moderate_corroboration_threshold": moderate_threshold,
            "required_cue_agreement": 2,
            "absolute_sample_jump_ratio_threshold": 2.0,
            "raw_jump_spectral_corroboration_threshold": (
                raw_jump_corroboration_threshold
            ),
            "merge_gap_seconds": merge_gap_seconds,
            "edge_guard_seconds": edge_guard_seconds,
            "frame_time_step_ms": round(hop / sample_rate * 1000.0, 3),
            "displayed_score_is_probability": False,
        },
        "features": {
            "duration_seconds": round(duration, 3),
            "number_of_feature_frames": int(len(starts)),
            "rms_db_min": round(float(np.min(rms_db)), 3),
            "rms_db_max": round(float(np.max(rms_db)), 3),
        },
        "findings": findings,
        "candidate_events_suppressed": len(suppressed_groups),
        "limitations": limitations,
    }