"""Standalone Streamlit dashboard for the Audio Forensics module."""
from __future__ import annotations

import hashlib
import html
import json
import base64
import math
import random

import matplotlib.pyplot as plt
import streamlit as st

from audio_forensics.analysis import analyze_decoded_audio
from audio_forensics.audio_io import AudioInputError, load_audio_bytes
from audio_forensics.evidence_store import store_evidence, verify_evidence
from audio_forensics.config import DEFAULT_SYNTHETIC_MODEL_ID, SUPPORTED_SUFFIXES
from audio_forensics.reports import generate_json_report, generate_pdf_report
from audio_forensics.visuals import waveform_figure, spectrogram_figure

st.set_page_config(page_title="Audio Forensics", page_icon="🎧", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1250px; }
    .hero { padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(135deg,#10243a,#1d4160); color: white; margin-bottom: 1.2rem; }
    .hero h1 { color: white; margin: 0 0 .35rem 0; letter-spacing: -.03em; }
    .hero p { color: #d8e7f5; margin: 0; }
    .small-note { color: #637083; font-size: .9rem; }
    div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.2); padding: .7rem; border-radius: 12px; }
</style>
<div class="hero">
  <h1>Audio Forensics</h1>
  <p>Evidence intake · Signal examination · Synthetic-speech screening · Reproducible reports</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# NEON AUDIO SPECTRUM BACKGROUND
# ---------------------------------------------------------
wave_rng = random.Random(42)
bars = []
trace_points = []

width = 1600
height = 900
center_y = 450

# Distinct waveform bursts produce a voice-spectrum silhouette.
bursts = [
    (75, 58), (250, 95), (450, 65), (670, 120),
    (900, 75), (1110, 110), (1335, 85), (1530, 65),
]

for x in range(0, width + 1, 8):
    envelope = max(
        math.exp(-((x - peak) / spread) ** 2)
        for peak, spread in bursts
    )

    amplitude = (
        8 + 125 * envelope * wave_rng.uniform(0.32, 1.0)
    )

    # Bright vertical spectrum bars.
    bars.append(
        f'<line x1="{x}" y1="{center_y-amplitude:.1f}" '
        f'x2="{x}" y2="{center_y+amplitude:.1f}" '
        f'stroke="url(#spectrum)" '
        f'stroke-width="{wave_rng.uniform(1.5, 3.2):.2f}" '
        f'opacity="{wave_rng.uniform(0.48, 0.92):.2f}"/>'
    )

    # Jagged signal trace laid over the frequency bars.
    sample = wave_rng.uniform(-1.0, 1.0) * amplitude
    trace_points.append((x, center_y + sample))

trace_path = "M " + " L ".join(
    f"{x} {y:.1f}" for x, y in trace_points
)

ambient_wave_svg = f"""
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 1600 900"
     preserveAspectRatio="none">
  <defs>
    <linearGradient id="spectrum" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ff3d71"/>
      <stop offset="22%" stop-color="#ff884d"/>
      <stop offset="45%" stop-color="#c026d3"/>
      <stop offset="70%" stop-color="#7958ff"/>
      <stop offset="100%" stop-color="#22d3ee"/>
    </linearGradient>

    <filter id="neon-glow" x="-20%" y="-80%"
            width="140%" height="260%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <g>
    {''.join(bars)}

    <path d="{trace_path}"
          fill="none"
          stroke="url(#spectrum)"
          stroke-width="7"
          opacity="0.22"
          filter="url(#neon-glow)"/>

    <path d="{trace_path}"
          fill="none"
          stroke="url(#spectrum)"
          stroke-width="2"
          stroke-linejoin="round"
          stroke-linecap="round"/>
  </g>
</svg>
"""

ambient_wave_data = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        ambient_wave_svg.encode("utf-8")
    ).decode("ascii")
)

ambient_wave_css = """
<style>
.stApp {
    position: relative;
    isolation: isolate;
    background:
        radial-gradient(
            ellipse at 50% 50%,
            rgba(124, 58, 237, 0.09),
            transparent 68%
        ),
        linear-gradient(135deg, #f8faff, #eff5ff, #fbf8ff)
        !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
header[data-testid="stHeader"] {
    background: transparent !important;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: -3vh -5vw;
    z-index: 0;
    pointer-events: none;

    background-image: url("__WAVE_IMAGE__");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 100% 100%;

    opacity: 0.9;
    transform-origin: center;
    animation: af-spectrum-motion 7s ease-in-out infinite alternate;
    will-change: transform;
}

@keyframes af-spectrum-motion {
    0% {
        transform: translate3d(-32px, -8px, 0) scaleY(0.94);
    }
    50% {
        transform: translate3d(0, 12px, 0) scaleY(1.06);
    }
    100% {
        transform: translate3d(32px, -6px, 0) scaleY(0.97);
    }
}

/* Keep the entire Streamlit application above the waveform. */
[data-testid="stAppViewContainer"] {
    position: relative !important;
    z-index: 1 !important;
    background: transparent !important;
}

[data-testid="stMain"] {
    position: relative !important;
    z-index: 1 !important;
    background: transparent !important;
}

.block-container {
    position: relative !important;
    z-index: 2 !important;
}

[data-testid="stSidebar"] {
    position: relative !important;
    z-index: 3 !important;
}

header[data-testid="stHeader"] {
    position: relative !important;
    z-index: 4 !important;
    background: transparent !important;
}

@media (prefers-reduced-motion: reduce) {
    .stApp::before {
        animation: none !important;
    }
}
</style>
"""

st.markdown(
    ambient_wave_css.replace("__WAVE_IMAGE__", ambient_wave_data),
    unsafe_allow_html=True,
)

st.caption("Prototype for forensic triage. Findings are indicators for human review, not proof that audio is authentic or manipulated.")

model_id = DEFAULT_SYNTHETIC_MODEL_ID

with st.sidebar:
    st.markdown("### Privacy")
    st.write(
        "Audio evidence is processed by the Python environment "
        "running this application. If hosted remotely, uploaded "
        "files are processed on that server. Do not upload sensitive "
        "police evidence to an unapproved deployment."
    )
    st.divider()
    st.markdown("### AI-generated voice screening")

    run_synthetic = st.checkbox(
        "Run experimental voice screening",
        value=False,
        help=(
            "Runs the lightweight AASIST-L ONNX detector on audio segments. "
            "Scores are experimental screening indicators, not proof of a deepfake."
        ),
    )

uploaded = st.file_uploader(
    "Upload one audio evidence file",
    type=[s.lstrip(".") for s in sorted(SUPPORTED_SUFFIXES)],
    help="WAV, FLAC and OGG are the most straightforward formats. Some compressed formats may require FFmpeg/librosa support.",
)

if uploaded is not None:
    raw = uploaded.getvalue()
    st.write(f"Selected file: **{uploaded.name}** · {len(raw):,} bytes")
    if st.button("Analyse audio", type="primary", use_container_width=True):
        try:
            with st.spinner("Preserving evidence and analysing audio..."):
                # Preserve the exact uploaded bytes before analysis.
                evidence_record = store_evidence(raw, uploaded.name)

                st.session_state["audio_evidence_store_record"] = (
                    evidence_record
                )

                # Decode a working copy for analysis.
                signal, meta = load_audio_bytes(raw, uploaded.name)

                # Confirm that storage and ingestion identify the same bytes.
                if evidence_record["sha256"] != meta["sha256"]:
                    raise ValueError(
                        "Stored evidence hash does not match the upload hash."
                    )

                result = analyze_decoded_audio(
                    raw,
                    uploaded.name,
                    signal,
                    meta,
                    include_synthetic_screening=run_synthetic,
                    model_id=model_id,
                )

                # Associate the analysis with its persistent evidence record.
                result["case_id"] = evidence_record["evidence_id"]

                st.session_state["audio_analysis_result"] = result
                st.session_state["audio_analysis_signal"] = signal
                st.session_state["audio_analysis_meta"] = meta
                st.session_state["audio_analysis_sample_rate"] = int(
                    meta["analysis_sample_rate_hz"]
                )
                st.session_state["audio_analysis_filename"] = uploaded.name

        except AudioInputError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")
            st.info(
                "Check the evidence-store files if preservation completed "
                "before the analysis failed."
            )

current_upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None
show_saved_result = (
    "audio_analysis_result" in st.session_state
    and (current_upload_hash is None or st.session_state["audio_analysis_result"].get("evidence", {}).get("sha256") == current_upload_hash)
)
if "audio_analysis_result" in st.session_state and not show_saved_result:
    st.info("A different file is selected. Click **Analyse audio** to replace the displayed results.")

if show_saved_result:
    result = st.session_state["audio_analysis_result"]
    signal = st.session_state["audio_analysis_signal"]
    sr = st.session_state["audio_analysis_sample_rate"]
    meta = st.session_state["audio_analysis_meta"]
    if result.get("evidence", {}).get("sha256") == meta.get("sha256"):
        st.success("Analysis completed for the currently stored evidence record.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{meta['decoded_duration_seconds']:.2f} s")
    c2.metric("Analysis sample rate", f"{sr:,} Hz")
    c3.metric("Channels analysed", "Mono")
    c4.metric("Edit candidates", str(len(result["analysis"]["signal_edit_screening"].get("findings", []))))

    with st.expander("Evidence identity and file metadata", expanded=True):
        st.json({k: v for k, v in meta.items() if k not in {"working_copy_modified", "original_bytes_modified"}})
        st.caption("The SHA-256 fingerprint refers to the exact uploaded bytes. Resampling and mono conversion are performed only on the analysis working copy.")
        stored_record = st.session_state.get("audio_evidence_store_record")
    current_hash = result.get("evidence", {}).get("sha256")

    st.divider()
    st.subheader("Persistent Evidence Record")

    if stored_record and stored_record.get("sha256") == current_hash:
        st.write("**Evidence ID:**", stored_record["evidence_id"])
        st.write("**Original filename:**", stored_record["original_filename"])
        st.write("**SHA-256:**", stored_record["sha256"])
        st.write("**File size:**", f'{stored_record["file_size_bytes"]:,} bytes')
        st.write("**Ingested at (UTC):**", stored_record["ingested_at_utc"])
        st.write("**Storage path:**", stored_record["stored_relative_path"])

        if st.button(
            "Verify stored original integrity",
            key=f"verify_{stored_record['evidence_id']}",
        ):
            try:
                verification = verify_evidence(stored_record["evidence_id"])

                if verification["status"] == "verified":
                    st.success("Integrity verified: stored bytes match the recorded SHA-256 and file size.")
                else:
                    st.error(f"Integrity check: {verification['status']}")

                st.json(verification)

            except Exception as exc:
                st.error(f"Verification failed: {exc}")

    else:
        st.warning(
            "No matching persistent evidence record found. "
            "Click Analyse audio to save this file into the evidence store."
        )

    st.caption(
        "A matching hash verifies the stored bytes against the recorded identity. "
        "It does not prove that the recording itself is authentic or unedited."
    )
    left, right = st.columns(2)
    # ---------------------------------------------------------
    # AI-GENERATED VOICE / ANTI-SPOOFING SCREENING
    # ---------------------------------------------------------
    synthetic = result.get("analysis", {}).get(
        "synthetic_speech_screening", {}
    )

    st.divider()
    st.subheader("AI-Generated Voice Screening")

    if synthetic.get("status") == "not_run":
        st.info(
            "Screening was not run for this analysis. "
            "Enable it in the sidebar and click Analyse audio again."
        )

    elif synthetic.get("status") == "inconclusive":
        st.warning(
            synthetic.get(
                "message",
                "The model could not produce a screening result.",
            )
        )

    elif synthetic.get("status") == "completed":
        st.caption(
            f"Model: {synthetic.get('model_id', 'Unknown')} | "
            f"Runtime: {synthetic.get('device', 'Unknown')}"
        )

        st.warning(
            "Experimental anti-spoofing results. A spoof-like score does "
            "not establish that audio is AI-generated, and a bona-fide-like "
            "score does not establish authenticity."
        )

        segments = synthetic.get("segments", [])

        if segments:
            rows = [
                {
                    "Segment": item["segment_id"],
                    "Start (s)": item["timestamp_start_seconds"],
                    "End (s)": item["timestamp_end_seconds"],
                    "Spoof-like score (%)": round(
                        item["model_score_fake_class"] * 100, 2
                    ),
                    "Bona-fide-like score (%)": round(
                        item["model_score_real_class"] * 100, 2
                    ),
                    "Model tendency": item["model_label"],
                }
                for item in segments
            ]

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"Unscored trailing audio: "
                f"{synthetic.get('unscored_tail_seconds', 0):.2f} seconds."
            )
        else:
            st.info("The model returned no scored segments.")

    else:
        st.error(
            f"Screening status: {synthetic.get('status', 'unknown')}"
        )
    with left:
        st.subheader("Waveform")
        st.caption(
            "How to read this: The blue line shows sound amplitude over time. "
            "Larger peaks generally indicate stronger sound; flatter regions "
            "may indicate pauses or quieter audio. Changes alone do not prove editing."
        )
        fig = waveform_figure(signal, sr)
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("Spectrogram")
        st.caption(
            "How to read this: Time runs from left to right, frequency runs "
            "from bottom to top, and brighter colours generally indicate "
            "stronger signal energy at those frequencies. Patterns are clues "
            "for investigation, not proof of manipulation."
        )
        fig = spectrogram_figure(signal, sr)
        st.pyplot(fig, clear_figure=True)

    # ---------------------------------------------------------
    # FORENSIC REVIEW: investigation timeline + candidate cards
    # ---------------------------------------------------------
    edit_result = result["analysis"]["signal_edit_screening"]
    findings = edit_result.get("findings", [])
    duration = max(float(len(signal) / sr), 0.001)

    evidence_hash = str(result.get("evidence", {}).get("sha256", "unknown"))
    playback_state_key = "audio_review_active_candidate"

    def format_time(seconds):
        total_ms = max(0, round(float(seconds) * 1000))
        minutes, remainder_ms = divmod(total_ms, 60_000)
        whole_seconds, milliseconds = divmod(remainder_ms, 1_000)
        return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

    def candidate_label(finding):
        labels = {
            "candidate_silence_transition": "Near-silence transition",
            "candidate_activity_transition": "Activity-level transition",
            "candidate_signal_discontinuity": "Signal discontinuity",
        }
        return labels.get(
            finding.get("finding_type"),
            "Signal-change candidate",
        )

    def review_label(finding):
        labels = {
            "higher": ("Elevated review priority", "#B54708"),
            "review": ("Review candidate", "#C77700"),
            "context": ("Context event", "#526581"),
        }
        return labels.get(
            finding.get("review_priority", "review"),
            ("Review candidate", "#C77700"),
        )

    # Visual styles for the timeline and finding cards.
    st.markdown(
        """
        <style>
        .af-panel-title {
            font-size: 1.35rem;
            font-weight: 750;
            color: #172B43;
            margin-bottom: .25rem;
        }
        .af-muted {
            color: #667085;
            font-size: .88rem;
        }
        .af-timeline-track {
            position: relative;
            height: 14px;
            border-radius: 30px;
            background: #E7ECF2;
            margin: 8px 0 10px;
        }
        .af-timeline-marker {
            position: absolute;
            top: -4px;
            width: 8px;
            height: 22px;
            transform: translateX(-50%);
            border: 1px solid #FFFFFF;
            border-radius: 4px;
            box-shadow: 0 1px 3px #00000020;
        }
        .af-timeline-axis {
            display: flex;
            justify-content: space-between;
            color: #667085;
            font-size: .76rem;
        }
        .af-status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: .78rem;
            font-weight: 650;
            background: #F2F4F7;
        }
        .af-section-label {
            color: #526581;
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .055em;
            margin-bottom: 5px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="af-panel-title">Investigation timeline</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "All timestamps are relative to the beginning of the uploaded recording. "
        "Select a finding below to listen to its surrounding audio."
    )

    # Summary metrics describe the current detector output, not probabilities.
    elevated_count = sum(
        f.get("review_priority") == "higher" for f in findings
    )
    context_count = sum(
        f.get("review_priority") == "context" for f in findings
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Candidate events", len(findings))
    m2.metric("Elevated review priority", elevated_count)
    m3.metric("Context events", context_count)

    # Each marker is positioned using the candidate peak timestamp.
    markers = []

    for finding in findings:
        peak = float(finding.get("peak_timestamp_seconds", 0))
        position = max(0.0, min(100.0, peak / duration * 100.0))

        _, marker_color = review_label(finding)
        finding_id = html.escape(
            str(finding.get("finding_id", "Candidate")),
            quote=True,
        )

        markers.append(
            f'<span class="af-timeline-marker" '
            f'style="left:{position:.3f}%;background:{marker_color}" '
            f'title="{finding_id} at {format_time(peak)}"></span>'
        )

    axis = "".join(
        f"<span>{format_time(duration * part)}</span>"
        for part in (0, 0.25, 0.50, 0.75, 1.0)
    )

    st.markdown(
        f"""
        <div style="padding:16px 4px 10px">
            <div class="af-timeline-axis">{axis}</div>
            <div class="af-timeline-track">
                {''.join(markers)}
            </div>
            <div class="af-muted">
                Timeline markers show where the detector found signal changes.
                They are not proof that an edit occurred.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="af-panel-title">Findings to review</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The relative score is a heuristic anomaly score, not a probability. "
        "Natural pauses, tone changes, background noise and codec artefacts "
        "may produce candidates. Smooth edits may go undetected."
    )

    if not findings:
        st.info(
            "No signal-change events crossed the screening thresholds. "
            "This does not establish that the recording is unedited."
        )

    else:
        for index, finding in enumerate(findings, start=1):
            finding_id = str(
                finding.get("finding_id", f"AUD-EDIT-{index:03d}")
            )

            start = float(finding.get("timestamp_start_seconds", 0))
            end = float(finding.get("timestamp_end_seconds", start))
            peak = float(finding.get("peak_timestamp_seconds", start))

            label = candidate_label(finding)
            priority, priority_color = review_label(finding)

            event_key = f"{evidence_hash}:{finding_id}"

            with st.container(border=True):
                # Header
                header_left, header_right = st.columns([3, 1.5])

                with header_left:
                    st.markdown(
                        f"### {format_time(start)} – {format_time(end)}"
                    )
                    st.caption(f"{finding_id}  ·  {label}")

                with header_right:
                    st.markdown(
                        f'<span class="af-status" '
                        f'style="color:{priority_color}">'
                        f'{html.escape(priority)}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Peak at {format_time(peak)}")

                st.markdown(
                    '<div class="af-section-label">What we observed</div>',
                    unsafe_allow_html=True,
                )

                event_type = finding.get("finding_type")

                observations = {
                    "candidate_silence_transition": (
                        "A transition between near-silence and active audio."
                    ),
                    "candidate_activity_transition": (
                        "A change between active audio and a quieter region."
                    ),
                    "candidate_signal_discontinuity": (
                        "Several signal-change indicators agree near this interval."
                    ),
                }

                st.write(
                    observations.get(
                        event_type,
                        "The signal screening identified an unusual interval.",
                    )
                )

                st.markdown(
                    '<div class="af-section-label">Why it was flagged</div>',
                    unsafe_allow_html=True,
                )

                st.write(
                    finding.get(
                        "explanation",
                        "The signal screening found a change that requires review.",
                    )
                )

                score = finding.get("score")
                if score is None:
                    score_text = "Not available"
                else:
                    try:
                        score_text = f"{float(score):.2f}"
                    except (TypeError, ValueError):
                        score_text = str(score)

                cue_count = finding.get(
                    "supporting_cue_count",
                    "Not reported",
                )

                st.caption(
                    f"Relative cue score: {score_text} — not a probability"
                    f"  ·  Supporting cues: {cue_count}"
                )

                button_label = "Listen to surrounding audio"

                if st.button(
                    button_label,
                    key=f"listen_{evidence_hash[:12]}_{index}",
                    use_container_width=False,
                ):
                    st.session_state[playback_state_key] = event_key

                if st.session_state.get(playback_state_key) == event_key:
                    # Listen to a few seconds on either side of the event.
                    clip_start = max(0.0, start - 3.0)
                    clip_end = min(duration, end + 3.0)

                    clip = signal[
                        int(clip_start * sr):int(clip_end * sr)
                    ]

                    st.caption(
                        f"Playback window: {format_time(clip_start)}"
                        f" – {format_time(clip_end)}"
                    )

                    if len(clip):
                        st.audio(clip, sample_rate=sr)

    with st.expander("Detection method and limitations"):
        for note in edit_result.get("limitations", []):
            st.write(f"- {note}")

    st.subheader("Export")
    export_left, export_right = st.columns(2)
    with export_left:
        try:
            pdf_bytes = generate_pdf_report(result, signal=signal, sample_rate=sr)
            st.download_button("Download audio forensic PDF", pdf_bytes, file_name="audio_forensic_report.pdf", mime="application/pdf", use_container_width=True)
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
    with export_right:
        st.download_button("Download structured JSON", generate_json_report(result), file_name="audio_forensic_report.json", mime="application/json", use_container_width=True)

    with st.expander("Integration output for the main application"):
        st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")
