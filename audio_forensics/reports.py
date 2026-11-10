"""PDF and JSON report generation for individual audio submissions."""
from __future__ import annotations

import io
import json
from html import escape
from typing import Any

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

from .visuals import waveform_figure, spectrogram_figure, figure_png_bytes


def _safe(value: Any) -> str:
    return escape(str(value if value is not None else "Not available"))


def _section(title: str, styles):
    return Paragraph(escape(title), styles["Heading2"])


def generate_pdf_report(result: dict[str, Any], *, signal: np.ndarray | None = None, sample_rate: int | None = None) -> bytes:
    """Create a standalone PDF report for one audio file; return PDF bytes."""
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title="Audio Forensic Examination Report",
        author="Audio Forensics Module",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallMuted", parent=styles["BodyText"], fontSize=8, textColor=colors.HexColor("#5B6573"), leading=10))
    styles.add(ParagraphStyle(name="FindingBody", parent=styles["BodyText"], fontSize=9, leading=12, spaceAfter=4))
    story = []
    story.append(Paragraph("Audio Forensic Examination", styles["Title"]))
    story.append(Paragraph("Digital evidence triage report · Human review required", styles["SmallMuted"]))
    story.append(Spacer(1, 6 * mm))

    evidence = result.get("evidence", {})
    analysis = result.get("analysis", {})
    metadata_rows = [
        ["Evidence file", _safe(evidence.get("filename"))],
        ["File size", f"{int(evidence.get('file_size_bytes') or 0):,} bytes"],
        ["SHA-256", _safe(evidence.get("sha256"))],
        ["Container / subtype", _safe(evidence.get("container_format")) + " / " + _safe(evidence.get("encoding_subtype"))],
        ["Original sample rate", _safe(evidence.get("source_sample_rate_hz")) + " Hz"],
        ["Analysis sample rate", _safe(evidence.get("analysis_sample_rate_hz")) + " Hz (mono working copy)"],
        ["Duration", f"{float(evidence.get('decoded_duration_seconds') or 0):.3f} seconds"],
        ["Analysis time (UTC)", _safe(result.get("created_at_utc"))],
    ]
    table = Table(metadata_rows, colWidths=[42 * mm, 133 * mm], repeatRows=0)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#182230")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([_section("1. Evidence identity", styles), table, Spacer(1, 5 * mm)])

    if signal is not None and sample_rate is not None:
        story.append(_section("2. Signal visualisation", styles))
        story.append(Paragraph("These plots describe the decoded working copy, not proof of manipulation.", styles["SmallMuted"]))
        wave_bytes = figure_png_bytes(waveform_figure(signal, sample_rate))
        spec_bytes = figure_png_bytes(spectrogram_figure(signal, sample_rate))
        story.append(Image(io.BytesIO(wave_bytes), width=172 * mm, height=44 * mm))
        story.append(Spacer(1, 2 * mm))
        story.append(Image(io.BytesIO(spec_bytes), width=172 * mm, height=54 * mm))
        story.append(Spacer(1, 4 * mm))
    else:
        story.append(_section("2. Signal visualisation", styles))
        story.append(Paragraph("Signal plots were not attached to this report generation request.", styles["SmallMuted"]))

    edits = analysis.get("signal_edit_screening", {})
    story.append(_section("3. Signal-change screening", styles))
    story.append(Paragraph(f"Method: {_safe(edits.get('method'))} · Status: {_safe(edits.get('status'))}", styles["FindingBody"]))
    edit_findings = edits.get("findings", [])
    if edit_findings:
        rows = [["Finding", "Interval (s)", "Relative score", "Observation"]]
        for finding in edit_findings:
            rows.append([
                _safe(finding.get("finding_id")),
                f"{float(finding.get('timestamp_start_seconds', 0)):.3f}–{float(finding.get('timestamp_end_seconds', 0)):.3f}",
                _safe(finding.get("score")),
                _safe(finding.get("explanation")),
            ])
        event_table = Table(rows, colWidths=[25 * mm, 30 * mm, 25 * mm, 92 * mm], repeatRows=1)
        event_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F365C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(event_table)
    else:
        story.append(Paragraph("No candidate discontinuities were flagged by the enabled heuristic. This does not prove the recording is unedited.", styles["FindingBody"]))

    for limitation in edits.get("limitations", []):
        story.append(Paragraph("• " + _safe(limitation), styles["SmallMuted"]))

    synthetic = analysis.get("synthetic_speech_screening", {})
    story.append(Spacer(1, 4 * mm))
    story.append(_section("4. Synthetic-speech screening", styles))
    story.append(Paragraph(f"Status: {_safe(synthetic.get('status'))}", styles["FindingBody"]))
    if synthetic.get("message"):
        story.append(Paragraph(_safe(synthetic.get("message")), styles["FindingBody"]))
    if synthetic.get("segments"):
        rows = [["Segment", "Interval (s)", "Model score: synthetic", "Model label"]]
        for segment in synthetic["segments"]:
            rows.append([
                _safe(segment.get("segment_id")),
                f"{float(segment.get('timestamp_start_seconds', 0)):.2f}–{float(segment.get('timestamp_end_seconds', 0)):.2f}",
                f"{float(segment.get('model_score_fake_class', 0)):.4f}",
                _safe(segment.get("model_label")),
            ])
        model_table = Table(rows, colWidths=[28 * mm, 38 * mm, 43 * mm, 63 * mm], repeatRows=1)
        model_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F365C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ]))
        story.append(model_table)
    for limitation in synthetic.get("limitations", []):
        story.append(Paragraph("• " + _safe(limitation), styles["SmallMuted"]))

    story.append(Spacer(1, 5 * mm))
    story.append(_section("5. Interpretation and limitations", styles))
    story.append(Paragraph(_safe(result.get("overall_interpretation")), styles["FindingBody"]))
    story.append(Paragraph(
        "This automated triage report is not a final forensic opinion. The original evidence should be preserved, "
        "methods and software versions should be recorded, and findings should be reviewed by a qualified investigator.",
        styles["SmallMuted"],
    ))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D8DEE8"))
        canvas.line(17 * mm, 12 * mm, A4[0] - 17 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#5B6573"))
        canvas.drawString(17 * mm, 7 * mm, "Audio Forensics Module · Triage only · Human review required")
        canvas.drawRightString(A4[0] - 17 * mm, 7 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def generate_json_report(result: dict[str, Any]) -> bytes:
    return json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
