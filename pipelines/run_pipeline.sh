#!/usr/bin/env bash
set -e

# 1. Add uv directory for Git Bash (/c/...) and WSL (/mnt/c/...)
export PATH="/c/Users/Lenovo/.local/bin:/mnt/c/Users/Lenovo/.local/bin:$HOME/.local/bin:$PATH"

# 2. Allow WSL or bash to call uv.exe directly if uv isn't aliased
if ! command -v uv &> /dev/null && command -v uv.exe &> /dev/null; then
    uv() { uv.exe "$@"; }
fi

echo "=== 1. PREPROCESSING ==="
uv run python scripts/preprocess.py --input data/titanic/raw/train.csv --output data/processed/train_cleaned.csv
uv run python scripts/preprocess.py --input data/titanic/raw/test.csv --output data/processed/test_cleaned.csv

echo "=== 2. FEATURIZING ==="
uv run python scripts/featurize.py --input data/processed/train_cleaned.csv --output data/features/train_features.csv
uv run python scripts/featurize.py --input data/processed/test_cleaned.csv --output data/features/test_features.csv

echo "=== 3. TRAINING ==="
uv run python scripts/train.py --input data/features/train_features.csv --output models/model.pkl

echo "=== 4. EVALUATING ==="
uv run python scripts/evaluate.py --model models/model.pkl --input data/features/train_features.csv --output metrics/eval.json

echo "=== 5. PREDICTING ==="
uv run python scripts/predict.py --model models/model.pkl --input data/features/test_features.csv --output predictions/predictions.csv

echo "=== PIPELINE COMPLETED SUCCESSFULLY ==="