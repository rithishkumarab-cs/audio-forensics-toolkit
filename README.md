# Audio Forensics & Evidence Integrity Toolkit

An independent Python-based engineering project exploring digital audio forensics, signal manipulation screening, synthetic-speech detection, and evidence integrity verification.

Built as a hands-on learning project inspired by real-world digital evidence authenticity challenges.

> **Project status:** Independent educational prototype. Not an officially commissioned, certified, or deployed forensic system.

## Overview

Digital audio can be modified, compressed, re-encoded, or synthetically generated. Examining such recordings requires multiple complementary techniques rather than relying on a single detection model.

This project brings together conventional digital signal processing, experimental AI-based synthetic-speech screening, cryptographic hashing, and structured forensic reporting in one interactive application.

The system is designed to help explore how audio evidence can be ingested, analysed, documented, and associated with integrity information.

## Features

- **Audio ingestion:** Accepts common audio formats and supported video containers containing audio streams.
- **Audio preprocessing:** Creates a mono, 16 kHz analysis working copy while retaining the original uploaded bytes.
- **Signal visualization:** Generates waveform and spectrogram visualizations.
- **Edit screening:** Uses RMS energy changes and spectral flux to identify candidate signal discontinuities.
- **Synthetic-speech screening:** Integrates the experimental AASIST-L model through ONNX Runtime.
- **Evidence integrity:** Uses SHA-256 hashes to identify and verify stored evidence bytes.
- **Evidence storage:** Maintains local evidence records and associated metadata.
- **Report generation:** Produces PDF and structured JSON analysis reports.
- **Interactive interface:** Provides audio upload, analysis, findings, visualization, and report downloads through Streamlit.

## Architecture

```text
              AUDIO / VIDEO UPLOAD
                       |
                       v
                 EVIDENCE INTAKE
                       |
                       v
               ORIGINAL SHA-256 HASH
                       |
                       v
                 AUDIO DECODING
                 Mono / 16 kHz
                       |
             +---------+---------+
             |                   |
             v                   v
       SIGNAL ANALYSIS    AI SCREENING
             |              (Optional)
             v                   v
       EDIT CANDIDATES    AASIST-L SCORES
             |                   |
             +---------+---------+
                       |
                       v
                ANALYSIS RESULTS
                       |
                       v
               PDF / JSON REPORTS
                       |
                       v
             EVIDENCE INTEGRATION
```

The original uploaded bytes are kept distinct from the decoded analysis signal. Analysis operates on the working copy, while evidence integrity is associated with the original upload.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Streamlit | Interactive application |
| Librosa | Audio processing |
| NumPy / SciPy | Numerical and signal processing |
| SoundFile | Audio file handling |
| Matplotlib | Waveform and spectrogram visualization |
| ONNX Runtime | AASIST-L model execution |
| SHA-256 | Evidence integrity hashing |
| ReportLab | PDF report generation |
| Pytest | Automated testing |

## Project Structure

```text
audio_forensics_module/
│
├── app.py
├── audio_forensics/
│   ├── analysis.py
│   ├── audio_io.py
│   ├── config.py
│   ├── edit_analysis.py
│   ├── evidence_store.py
│   ├── reports.py
│   ├── synthetic_detector.py
│   └── visuals.py
│
├── models/                 # Local model weights; not committed
├── docs/
│   ├── integration.md
│   └── test-plan.md
├── scripts/
│   ├── evaluate_edit_detector.py
│   └── generate_demo_audio.py
├── tests/
├── requirements.txt
├── requirements-ai.txt
├── requirements-dev.txt
├── setup_windows.ps1
├── run_windows.ps1
└── .gitignore
```

## Installation

### Prerequisites

- Python 3.12 recommended.
- Git.
- Windows, Linux, or another supported Python environment.

### 1. Clone the repository

```powershell
git clone <REPOSITORY_URL>
cd audio_forensics_module
```

### 2. Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

ONNX Runtime is currently required by the analysis module's import chain, even when synthetic-speech screening is not selected. The downloaded model weights themselves are optional.

### 5. Run the application

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Synthetic-Speech Screening

The experimental AI screening component uses **AASIST-L**, executed through ONNX Runtime.

To enable it:

1. Install the optional requirements.
2. Obtain the compatible AASIST-L ONNX model from its trusted upstream source.
3. Verify the upstream model's licence and redistribution terms.
4. Place the model at:

```text
models/aasist-l.onnx
```

5. Enable experimental voice screening in the application sidebar.

The model binary is intentionally excluded from this repository.

### Model limitations

- Output scores are model scores, not calibrated probabilities.
- A spoof-like score does not establish that audio is AI-generated.
- A bona-fide-like score does not prove that a recording is genuine.
- Performance may vary across languages, accents, codecs, recording conditions, and unseen speech generators.
- Model output should be treated as an experimental screening signal requiring independent review.

## Evidence Integrity

The project uses SHA-256 hashing to associate an uploaded file with its stored evidence record.

The application compares hashes obtained during evidence storage and audio ingestion to check that the same original bytes are being referenced.

A matching hash supports byte-level integrity verification. It does not independently establish the source, creation time, provenance, or original authenticity of a recording.

## Signal-Level Analysis

The edit-screening component examines signal characteristics including:

- Root Mean Square (RMS) energy.
- Spectral flux.
- Candidate discontinuities and associated timestamps.

These measurements can identify regions that deserve closer examination. Natural pauses, changes in volume, noise, compression, and recording conditions can also produce similar signal characteristics.

Detected candidates are therefore not confirmed evidence of manipulation.

## Reports and Integration

The application generates:

- PDF forensic analysis reports.
- Structured JSON analysis results.
- Evidence metadata and SHA-256 identifiers.
- Timestamp-associated findings.
- Explicit interpretation and limitation information.

The module exposes integration functions through the `audio_forensics` Python package.

Example:

```python
from audio_forensics import analyze_audio, generate_pdf_report

result = analyze_audio(
    uploaded_bytes,
    "recording.wav",
    include_synthetic_screening=False,
)
```

The host application is responsible for its own case management, authorization, access controls, and audit policies.

Refer to `docs/integration.md` for the integration details.

## Testing

Run the test suite:

```powershell
python -m pytest -q
```

Previously verified test baseline:

```text
35 passed
```

Run a public API import check:

```powershell
python -c "from audio_forensics import analyze_audio, analyze_decoded_audio, generate_pdf_report; print('Public API imports OK')"
```

Refer to `docs/test-plan.md` for the test methodology.

## Security and Limitations

This project is an experimental educational prototype, not a validated forensic instrument.

It does not provide definitive authentication of recordings, guaranteed deepfake detection, production-grade evidence vaulting, or complete chain-of-custody management.

Do not upload confidential investigative recordings to an unapproved deployment. Additional access controls, encryption, audit logging, evidence-management procedures, and independent validation would be necessary before operational use.

## Project Direction

The project is intended to evolve through practical engineering work:

- Improve signal-analysis algorithms and evaluation.
- Strengthen evidence integrity and auditability.
- Improve synthetic-speech screening and out-of-domain evaluation.
- Explore integration with image and video forensic modules.
- Develop a unified digital evidence analysis workflow.

**Built independently as a software engineering and digital forensics learning project.**