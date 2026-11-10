import io
import numpy as np
import soundfile as sf
import pytest

from audio_forensics.audio_io import load_audio_bytes, AudioInputError, sha256_hex


def make_wav(samples, sr=16000):
    b = io.BytesIO()
    sf.write(b, samples, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def test_load_wav_preserves_original_hash_and_resamples():
    sr = 22050
    t = np.arange(sr, dtype=np.float32) / sr
    samples = 0.15 * np.sin(2 * np.pi * 440 * t)
    raw = make_wav(samples, sr)
    signal, meta = load_audio_bytes(raw, "sample.wav")
    assert meta["sha256"] == sha256_hex(raw)
    assert meta["source_sample_rate_hz"] == sr
    assert meta["analysis_sample_rate_hz"] == 16000
    assert meta["original_bytes_modified"] is False
    assert len(signal) == pytest.approx(16000, abs=2)


def test_empty_upload_rejected():
    with pytest.raises(AudioInputError):
        load_audio_bytes(b"", "empty.wav")


def test_unsupported_extension_rejected():
    with pytest.raises(AudioInputError):
        load_audio_bytes(b"not audio", "sample.exe")


def _ffmpeg_exe_for_test():
    import shutil
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def test_mp4_audio_stream_is_decoded_and_original_hash_preserved(tmp_path):
    import subprocess

    ffmpeg = _ffmpeg_exe_for_test()
    if not ffmpeg:
        pytest.skip("FFmpeg is not available in this test environment")

    mp4_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    original = mp4_path.read_bytes()
    signal, meta = load_audio_bytes(original, "sample.mp4")

    assert meta["sha256"] == sha256_hex(original)
    assert meta["original_bytes_modified"] is False
    assert meta["decoder_used"] == "ffmpeg"
    assert meta["analysis_sample_rate_hz"] == 16000
    assert meta["analysis_channels"] == 1
    assert len(signal) > 30_000
    assert len(signal) == pytest.approx(32_000, abs=1_600)
    assert meta["container_format"] != "not identified by libsndfile"


def test_video_container_without_audio_returns_clear_error(tmp_path):
    import subprocess

    ffmpeg = _ffmpeg_exe_for_test()
    if not ffmpeg:
        pytest.skip("FFmpeg is not available in this test environment")

    mp4_path = tmp_path / "video_without_audio.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
            "-an", "-c:v", "mpeg4", str(mp4_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    with pytest.raises(AudioInputError, match="does not contain a readable audio stream"):
        load_audio_bytes(mp4_path.read_bytes(), "video_without_audio.mp4")


def test_mp4_video_with_audio_stream_is_decoded(tmp_path):
    import subprocess

    ffmpeg = _ffmpeg_exe_for_test()
    if not ffmpeg:
        pytest.skip("FFmpeg is not available in this test environment")

    mp4_path = tmp_path / "video_with_audio.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(mp4_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    original = mp4_path.read_bytes()
    signal, meta = load_audio_bytes(original, "video_with_audio.mp4")

    assert meta["sha256"] == sha256_hex(original)
    assert meta["decoder_used"] == "ffmpeg"
    assert meta["source_sample_rate_hz"] == 44100
    assert meta["analysis_sample_rate_hz"] == 16000
    assert meta["original_bytes_modified"] is False
    assert 1.9 <= meta["decoded_duration_seconds"] <= 2.1
    assert signal.dtype == np.float32
