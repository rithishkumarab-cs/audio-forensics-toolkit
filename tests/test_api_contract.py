import io
import json
import numpy as np
import soundfile as sf

from audio_forensics.analysis import analyze_audio
from audio_forensics.audio_io import load_audio_bytes
from audio_forensics.reports import generate_json_report, generate_pdf_report


def make_tone_wav(duration=3.5, sr=16000):
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    x = (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV")
    return b.getvalue(), x


def test_service_result_is_json_serializable_and_separates_synthetic_screening():
    raw, _ = make_tone_wav()
    result = analyze_audio(raw, "fixture.wav")
    json.loads(generate_json_report(result).decode("utf-8"))
    assert result["module"] == "audio"
    assert result["analysis"]["synthetic_speech_screening"]["status"] == "not_run"
    assert result["evidence"]["sha256"]


def test_pdf_report_starts_with_pdf_signature():
    raw, _ = make_tone_wav()
    signal, metadata = load_audio_bytes(raw, "fixture.wav")
    result = analyze_audio(raw, "fixture.wav")
    pdf = generate_pdf_report(result, signal=signal, sample_rate=metadata["analysis_sample_rate_hz"])
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 10_000
