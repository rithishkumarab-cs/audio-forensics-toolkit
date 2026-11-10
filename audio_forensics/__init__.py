"""Audio forensics component for integration into a larger investigation platform."""

from .analysis import analyze_audio, analyze_decoded_audio
from .reports import generate_pdf_report

__all__ = ["analyze_audio", "analyze_decoded_audio", "generate_pdf_report"]
