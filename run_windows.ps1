$ErrorActionPreference = 'Stop'
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run .\setup_windows.ps1 first."
}
& $python -m streamlit run app.py
