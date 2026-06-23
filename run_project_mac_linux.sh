#!/usr/bin/env bash
set -e

echo "========================================================"
echo "GridLock R2 - Full Pipeline (Train from Scratch)"
echo "========================================================"
echo ""

echo "[1/3] Creating Python Virtual Environment..."
if [ ! -f "venv/bin/activate" ]; then
    python3 -m venv venv || python -m venv venv
fi

echo "[2/3] Activating venv and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --no-cache-dir
pip install -r requirements.txt --no-cache-dir

echo "[3/4] Running End-to-End Pipeline (Training from Scratch)..."
python -m src.data.pipeline

echo ""
echo "[4/4] Generating SHAP Feature Importance Analysis..."
python src/evaluation/shap_analysis.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[WARNING] SHAP analysis failed. Dashboard will show empty feature importance chart."
fi

echo ""
echo "========================================================"
echo "[SUCCESS] Pipeline completed successfully."
echo "========================================================"
echo ""
echo "[5/5] Starting servers for demo..."
uvicorn src.api.main:app --host 127.0.0.1 --port 9000 > api.log 2>&1 &
API_PID=$!
python -m http.server 9090 --directory docs > frontend.log 2>&1 &
FRONTEND_PID=$!

echo "API Backend: http://127.0.0.1:9000"
echo "Dashboard:   http://localhost:9090"
echo "Press Ctrl+C to stop both servers."

trap "echo 'Shutting down servers...'; kill $API_PID $FRONTEND_PID; exit" SIGINT SIGTERM
wait
