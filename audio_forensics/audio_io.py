"""Audio ingestion helpers that preserve original bytes and analyse a decoded copy.

For video containers (MP4/MOV/MKV/etc.) FFmpeg extracts the first audio stream into
an in-memory mono float32 working signal. When a seekable input is required, the
uploaded bytes are written to a short-lived temporary file that is removed on exit.
The SHA-256 fingerprint is always calculated from the exact uploaded bytes.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import (
    DEFAULT_SAMPLE_RATE,
    MAX_DURATION_SECONDS,
    MAX_UPLOAD_BYTES,
    SUPPORTED_SUFFIXES,
    VIDEO_CONTAINER_SUFFIXES,
)


class AudioInputError(ValueError):
    """Raised when uploaded audio cannot safely be analysed."""


def sha256_hex(data: bytes) -> str:
    """Return SHA-256 for the exact bytes received, before any decoding/conversion."""
    return hashlib.sha256(data).hexdigest()


def _metadata_with_soundfile(data: bytes) -> dict[str, Any]:
    try:
        info = sf.info(io.BytesIO(data))
        duration = float(info.duration) if info.samplerate else None
        return {
            "container_format": str(info.format or "unknown"),
            "encoding_subtype": str(info.subtype or "unknown"),
            "sample_rate_hz": int(info.samplerate) if info.samplerate else None,
            "channels": int(info.channels) if info.channels else None,
            "frames": int(info.frames) if info.frames else None,
            "duration_seconds": duration,
        }
    except Exception:
        return {
            "container_format": "not identified by libsndfile",
            "encoding_subtype": "unknown",
            "sample_rate_hz": None,
            "channels": None,
            "frames": None,
            "duration_seconds": None,
        }


def _decode_with_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if samples.size == 0:
        raise AudioInputError("The uploaded file contains no decodable audio samples.")
    mono = np.mean(samples, axis=1, dtype=np.float32)
    return mono.astype(np.float32, copy=False), int(sample_rate)


def _decode_with_librosa(data: bytes) -> tuple[np.ndarray, int]:
    """Fallback for compressed audio supported by the local librosa decoder stack."""
    import librosa  # optional decoder fallback; imported only when needed

    signal, sample_rate = librosa.load(io.BytesIO(data), sr=None, mono=True)
    if signal.size == 0:
        raise AudioInputError("The uploaded file contains no decodable audio samples.")
    return np.asarray(signal, dtype=np.float32), int(sample_rate)


def _get_ffmpeg_exe() -> str | None:
    """Prefer a system FFmpeg; otherwise use the executable bundled by imageio-ffmpeg."""
    system_exe = shutil.which("ffmpeg")
    if system_exe:
        return system_exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _probe_media(path: Path, ffmpeg_exe: str) -> dict[str, Any]:
    """Best-effort metadata probe. FFprobe is optional; decoding still works without it."""
    ffprobe_exe = shutil.which("ffprobe")
    if ffprobe_exe:
        try:
            completed = subprocess.run(
                [
                    ffprobe_exe,
                    "-v", "error",
                    "-show_streams",
                    "-show_format",
                    "-of", "json",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
            if completed.returncode == 0:
                parsed = json.loads(completed.stdout.decode("utf-8", errors="replace"))
                streams = parsed.get("streams", [])
                audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
                fmt = parsed.get("format", {})
                if audio_stream is None:
                    return {
                        "_has_audio_stream": False,
                        "container_format": fmt.get("format_name") or path.suffix.lstrip(".").upper(),
                        "encoding_subtype": "unknown",
                    }
                try:
                    sample_rate = int(audio_stream.get("sample_rate"))
                except (TypeError, ValueError):
                    sample_rate = None
                try:
                    channels = int(audio_stream.get("channels"))
                except (TypeError, ValueError):
                    channels = None
                try:
                    duration = float(audio_stream.get("duration") or fmt.get("duration"))
                except (TypeError, ValueError):
                    duration = None
                try:
                    frames = int(audio_stream.get("nb_frames"))
                except (TypeError, ValueError):
                    frames = None
                return {
                    "_has_audio_stream": True,
                    "container_format": fmt.get("format_name") or path.suffix.lstrip(".").upper(),
                    "encoding_subtype": audio_stream.get("codec_name") or "unknown",
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    # For compressed streams, nb_frames may describe codec packets rather
                    # than decoded PCM samples; do not claim it is a PCM frame count.
                    "frames": frames,
                    "duration_seconds": duration,
                }
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

    # Without FFprobe, FFmpeg's input summary can still supply best-effort metadata.
    try:
        completed = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-i", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
        summary = completed.stderr.decode("utf-8", errors="replace")
        audio_lines = [line for line in summary.splitlines() if "Audio:" in line]
        if not audio_lines:
            return {
                "_has_audio_stream": False,
                "container_format": path.suffix.lstrip(".").upper() or "unknown",
                "encoding_subtype": "unknown",
            }
        line = audio_lines[0]
        rate_match = re.search(r"\b(\d{4,6})\s*Hz\b", line)
        channel_match = re.search(r"\b(mono|stereo|\d+(?:\.\d+)?\s+channels)\b", line, re.IGNORECASE)
        sample_rate = int(rate_match.group(1)) if rate_match else None
        channels = None
        if channel_match:
            channel_text = channel_match.group(1).lower()
            if channel_text == "mono":
                channels = 1
            elif channel_text == "stereo":
                channels = 2
            else:
                number_match = re.search(r"\d+", channel_text)
                channels = int(number_match.group(0)) if number_match else None
        codec_match = re.search(r"Audio:\s*([^,\s]+)", line)
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", summary)
        duration = None
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return {
            "_has_audio_stream": True,
            "container_format": path.suffix.lstrip(".").upper() or "unknown",
            "encoding_subtype": codec_match.group(1) if codec_match else "unknown",
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "frames": None,
            "duration_seconds": duration,
        }
    except (OSError, subprocess.TimeoutExpired):
        return {
            "_has_audio_stream": None,
            "container_format": path.suffix.lstrip(".").upper() or "unknown",
            "encoding_subtype": "unknown",
            "sample_rate_hz": None,
            "channels": None,
            "frames": None,
            "duration_seconds": None,
        }


def _decode_with_ffmpeg(
    data: bytes,
    filename: str,
    target_sample_rate: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract the first audio stream using FFmpeg into a mono float32 working signal."""
    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe:
        raise AudioInputError(
            "This format needs FFmpeg decoding, but FFmpeg is unavailable. "
            "Run the project setup script to install the bundled FFmpeg dependency, then retry."
        )

    suffix = PurePath(filename).suffix.lower()
    # Use a short-lived, seekable file because some MP4/MOV files cannot be safely probed
    # when supplied as a non-seekable stdin stream. The temporary directory is cleaned up.
    try:
        with tempfile.TemporaryDirectory(prefix="audio_forensics_") as temp_dir:
            input_path = Path(temp_dir) / f"evidence{suffix}"
            input_path.write_bytes(data)
            probed = _probe_media(input_path, ffmpeg_exe)

            if probed.get("_has_audio_stream") is False:
                raise AudioInputError(
                    "This media file does not contain a readable audio stream. "
                    "Please upload a file with an audio track."
                )

            source_duration = probed.get("duration_seconds")
            if source_duration is not None and source_duration > MAX_DURATION_SECONDS:
                raise AudioInputError(
                    f"Recording is {source_duration:.1f} seconds long. The prototype currently supports "
                    f"up to {MAX_DURATION_SECONDS} seconds per analysis."
                )

            command = [
                ffmpeg_exe,
                "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(input_path),
                "-map", "0:a:0",
                "-vn", "-sn", "-dn",
                # Decode at most one second over the cap so long files can't allocate
                # unbounded PCM output before the post-decode duration validation.
                "-t", str(MAX_DURATION_SECONDS + 1),
                "-ac", "1",
                "-ar", str(target_sample_rate),
                "-c:a", "pcm_f32le",
                "-f", "f32le",
                "pipe:1",
            ]
            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=120,
                )
            except subprocess.TimeoutExpired as exc:
                raise AudioInputError(
                    "Audio extraction exceeded the processing time limit. "
                    "Try a shorter recording or a supported audio-only file."
                ) from exc

            if completed.returncode != 0 or not completed.stdout:
                error_text = completed.stderr.decode("utf-8", errors="replace").lower()
                if "matches no streams" in error_text or "does not contain any stream" in error_text:
                    raise AudioInputError(
                        "This media file does not contain a readable audio stream. "
                        "Please upload a file with an audio track."
                    )
                raise AudioInputError(
                    "FFmpeg could not decode the audio from this file. It may be corrupted, "
                    "encrypted, unsupported by the available codecs, or missing an audio track."
                )

            if len(completed.stdout) % 4 != 0:
                raise AudioInputError("FFmpeg returned an incomplete audio sample stream.")
            signal = np.frombuffer(completed.stdout, dtype="<f4").copy()
            if signal.size == 0:
                raise AudioInputError("The uploaded media contains no decodable audio samples.")

            decoded_duration = signal.size / float(target_sample_rate)
            if decoded_duration > MAX_DURATION_SECONDS:
                raise AudioInputError(
                    f"Recording exceeds the {MAX_DURATION_SECONDS}-second analysis limit."
                )

            # Drop the private helper key and retain only report-safe metadata.
            probed.pop("_has_audio_stream", None)
            metadata = probed
            return signal.astype(np.float32, copy=False), metadata
    except AudioInputError:
        raise
    except OSError as exc:
        raise AudioInputError("The uploaded file could not be staged for safe decoding.") from exc


def load_audio_bytes(
    data: bytes,
    filename: str,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate uploaded media, fingerprint original bytes, decode and resample a working copy.

    The returned signal is mono float32 at target_sample_rate. The SHA-256 fingerprint
    always refers to the exact original uploaded bytes, before any decoding/conversion.
    """
    if not data:
        raise AudioInputError("No file data was received.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AudioInputError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")

    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise AudioInputError(
            f"Unsupported file extension '{suffix or '(none)'}'. Supported formats: "
            + ", ".join(sorted(SUPPORTED_SUFFIXES))
        )

    metadata = _metadata_with_soundfile(data)
    decoder_used = "soundfile"
    already_at_target_rate = False
    source_sr: int | None = None

    # Video containers commonly cannot be decoded by libsndfile. Use FFmpeg directly
    # so MP4/MOV/MKV uploads do not depend on the user's VLC conversion workflow.
    if suffix in VIDEO_CONTAINER_SUFFIXES:
        signal, ffmpeg_meta = _decode_with_ffmpeg(data, filename, target_sample_rate)
        metadata.update({k: v for k, v in ffmpeg_meta.items() if v is not None})
        source_sr = ffmpeg_meta.get("sample_rate_hz")
        decoder_used = "ffmpeg"
        already_at_target_rate = True
    else:
        try:
            signal, source_sr = _decode_with_soundfile(data)
        except Exception as soundfile_error:
            try:
                signal, source_sr = _decode_with_librosa(data)
                decoder_used = "librosa"
            except Exception:
                try:
                    signal, ffmpeg_meta = _decode_with_ffmpeg(data, filename, target_sample_rate)
                    metadata.update({k: v for k, v in ffmpeg_meta.items() if v is not None})
                    source_sr = ffmpeg_meta.get("sample_rate_hz")
                    decoder_used = "ffmpeg"
                    already_at_target_rate = True
                except AudioInputError:
                    raise
                except Exception as exc:
                    raise AudioInputError(
                        "Could not decode this recording. Verify that it is not corrupted and that "
                        "the required codecs are available."
                    ) from exc

    # For FFmpeg output the sample rate was explicitly set to target_sample_rate.
    signal_sample_rate = target_sample_rate if already_at_target_rate else source_sr
    if signal_sample_rate is None or signal_sample_rate <= 0:
        raise AudioInputError("The decoded audio has an unknown or invalid sample rate.")
    if not np.all(np.isfinite(signal)):
        raise AudioInputError("The decoded audio contains NaN or infinite samples.")

    signal = np.clip(signal, -1.0, 1.0).astype(np.float32, copy=False)
    if signal_sample_rate != target_sample_rate:
        from math import gcd
        factor = gcd(int(signal_sample_rate), int(target_sample_rate))
        up = target_sample_rate // factor
        down = signal_sample_rate // factor
        signal = resample_poly(signal, up, down).astype(np.float32, copy=False)

    duration = float(len(signal) / target_sample_rate)
    if duration <= 0:
        raise AudioInputError("The recording duration is zero.")
    if duration > MAX_DURATION_SECONDS:
        raise AudioInputError(
            f"Recording is {duration:.1f} seconds long. The prototype currently supports up to "
            f"{MAX_DURATION_SECONDS} seconds per analysis."
        )

    metadata.update({
        "filename": PurePath(filename).name,
        "file_size_bytes": len(data),
        "sha256": sha256_hex(data),
        "source_sample_rate_hz": source_sr,
        "analysis_sample_rate_hz": target_sample_rate,
        "analysis_channels": 1,
        "decoded_duration_seconds": duration,
        "decoder_used": decoder_used,
        "working_copy_modified": True,
        "original_bytes_modified": False,
    })
    return signal, metadata
