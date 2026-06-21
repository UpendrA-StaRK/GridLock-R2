#!/usr/bin/env bash
set -e

echo "========================================================"
echo "GridLock R2 - Inference Only (Uses Existing Checkpoint)"
echo "========================================================"
echo ""

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found. Please run run_project.sh first to set up the environment."
    exit 1
fi

echo "[1/2] Activating venv..."
source venv/bin/activate

echo "[2/2] Running End-to-End Pipeline (Skipping Training)..."
python -m src.data.pipeline --skip-training --ckpt-dir submission_checkpoint

echo ""
echo "========================================================"
echo "[SUCCESS] Inference completed successfully."
echo "========================================================"
