#!/usr/bin/env bash
# run_pipeline.sh
# ----------------
# Runs the full pipeline end-to-end:
#   1. Generate synthetic data (skip if data/raw already has your own images)
#   2. Preprocess (denoise/deskew/crop)
#   3. Train all 4 models
#   4. Evaluate + compare all 4 models
#
# Usage:
#   bash run_pipeline.sh              # full run with pretrained transfer models
#   bash run_pipeline.sh --no-pretrained --epochs 5 --per-class 40 --cpu

set -e

# ── Auto-detect venv Python on Windows ──────────────────────────────
# When 'bash' is called from PowerShell, it does NOT inherit the venv.
# We look for the venv Python relative to this script's directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/venv/Scripts/python.exe" ]; then
  PYTHON="$SCRIPT_DIR/venv/Scripts/python.exe"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
  PYTHON="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON="python"
fi

echo "Using Python: $PYTHON"

EPOCHS=8
PER_CLASS=60
EXTRA_TRAIN_ARGS=""
FORCE_CPU_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pretrained) EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --no_pretrained"; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --per-class) PER_CLASS="$2"; shift 2 ;;
    --cpu) FORCE_CPU_FLAG="1"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

export FORCE_CPU="$FORCE_CPU_FLAG"

echo "==> Step 0/4: Checking dependencies"
if ! "$PYTHON" -c "import cv2, torch, torchvision, fastapi" 2>/dev/null; then
  echo ""
  echo "ERROR: Required Python packages are missing."
  echo "Fix: install the requirements into your venv:"
  echo "    python -m pip install -r requirements.txt"
  exit 1
fi

echo "==> Step 1/4: Generating synthetic data (skip if data/raw/ already populated)"
if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
  "$PYTHON" data/generate_synthetic_data.py --per_class "$PER_CLASS"
else
  echo "data/raw already has files -- skipping synthetic generation."
fi

echo "==> Step 2/4: Preprocessing (denoise / deskew / crop)"
"$PYTHON" src/data_preprocessing.py

echo "==> Step 3/4: Training all models ($EPOCHS epochs each)"
"$PYTHON" -m src.train --model all --epochs "$EPOCHS" $EXTRA_TRAIN_ARGS

echo "==> Step 4/4: Evaluating and comparing all models"
"$PYTHON" -m src.evaluate

echo ""
echo "Pipeline complete."
echo "  - Trained checkpoints : outputs/models/"
echo "  - Comparison table    : outputs/reports/model_comparison.md"
echo "  - Confusion matrices  : outputs/reports/confusion_matrix_*.png"
echo ""
echo "Start the API with:  uvicorn api.main:app --reload --port 8000"
