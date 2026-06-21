#!/usr/bin/env bash
cd "$(dirname "$0")/.."
set -e

echo "========================================================"
echo "GridLock R2 - Fresh System Setup & Execution"
echo "========================================================"
echo ""

echo "[1/3] Creating Python Virtual Environment..."
python3 -m venv venv || python -m venv venv

echo "[2/3] Activating venv and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --no-cache-dir
pip install -r requirements.txt --no-cache-dir

echo "[3/3] Running End-to-End Pipeline (Training from Scratch)..."
python -m src.data.pipeline

echo ""
echo "========================================================"
echo "[SUCCESS] Pipeline completed successfully."
echo "========================================================"
