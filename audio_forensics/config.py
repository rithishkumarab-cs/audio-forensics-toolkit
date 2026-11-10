"""Configuration defaults. Keep model loading optional and explicit."""
from __future__ import annotations

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_SYNTHETIC_MODEL_ID = "SpeechAntiSpoofingBenchmarks/AASIST-L"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_DURATION_SECONDS = 300            # 5 minutes for the hackathon prototype

# Audio-only files plus common video containers that may contain an audio stream.
# FFmpeg is used to extract the audio stream from the latter formats.
SUPPORTED_SUFFIXES = {
    ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".aiff", ".aif",
    ".wma", ".webm", ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".mpeg",
    ".mpg", ".3gp", ".ts", ".m2ts", ".flv", ".wmv", ".ogv",
}
VIDEO_CONTAINER_SUFFIXES = {
    ".webm", ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".mpeg", ".mpg",
    ".3gp", ".ts", ".m2ts", ".flv", ".wmv", ".ogv",
}
MODEL_WINDOW_SECONDS = 4.0
MODEL_MIN_WINDOW_SECONDS = 2.5
