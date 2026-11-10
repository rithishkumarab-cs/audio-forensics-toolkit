$ErrorActionPreference = 'Stop'

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}
$python = ".\.venv\Scripts\python.exe"
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
Write-Host "Installing core requirements..." -ForegroundColor Cyan
& $python -m pip install -r requirements.txt
Write-Host "Core setup complete. Run: .\.venv\Scripts\python.exe -m streamlit run app.py" -ForegroundColor Green
Write-Host "Optional AI detector: install a compatible PyTorch build first, then pip install -r requirements-ai.txt" -ForegroundColor Yellow
