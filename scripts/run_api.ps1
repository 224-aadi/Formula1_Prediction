# scripts/run_api.ps1
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location -Path $rootDir

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    .\.venv\Scripts\Activate.ps1
} elseif (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Virtual environment not found! Please run 'python -m venv .venv' and 'pip install -e .'" -ForegroundColor Yellow
}

$env:F1_DATASET_PATH="data/processed/final_hybrid_dataset.parquet"
$env:F1_RANKER_PATH="artifacts/final/ranker.joblib"
$env:F1_DNF_RAW_PATH="artifacts/final/dnf_raw.joblib"
$env:F1_DNF_CAL_PATH="artifacts/final/dnf_cal.joblib"

Write-Host "Starting Uvicorn API server..." -ForegroundColor Green
python -m uvicorn f1outcome.api.app:app --reload --app-dir src
