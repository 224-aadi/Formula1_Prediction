#!/bin/bash
set -e

# Change to project root directory
cd "$(dirname "$0")/.."

if [ -f "./.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ./.venv/bin/activate
elif [ -f "./venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ./venv/bin/activate
else
    echo "Virtual environment not found! Please run 'python -m venv .venv' and 'pip install -e .'"
fi

export F1_DATASET_PATH="data/processed/final_hybrid_dataset.parquet"
export F1_RANKER_PATH="artifacts/final/ranker.joblib"
export F1_DNF_RAW_PATH="artifacts/final/dnf_raw.joblib"
export F1_DNF_CAL_PATH="artifacts/final/dnf_cal.joblib"

echo "Starting Uvicorn API server..."
python -m uvicorn f1outcome.api.app:app --reload --app-dir src
