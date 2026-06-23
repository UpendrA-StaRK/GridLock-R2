#!/bin/bash
set -e

echo "========================================================"
echo "GridLock R2 - Hackathon Demo Launcher (Instant Inference)"
echo "========================================================"
echo ""

if [ ! -f "venv/bin/activate" ]; then
    echo "[1/3] Virtual environment not found. Creating and installing dependencies..."
    python3 -m venv venv || python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "[1/3] Activating venv..."
    source venv/bin/activate
fi

echo "[2/4] Generating SHAP Feature Importance Analysis..."
python src/evaluation/shap_analysis.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[WARNING] SHAP analysis failed. Dashboard will show empty feature importance chart."
fi

echo ""
echo "[3/4] Starting FastAPI Backend Server on port 9000..."
uvicorn src.api.main:app --host 127.0.0.1 --port 9000 > api.log 2>&1 &
API_PID=$!

echo "[4/4] Starting Frontend Server on port 9090..."
python -m http.server 9090 --directory docs > frontend.log 2>&1 &
FRONTEND_PID=$!

echo ""
echo "========================================================"
echo "[SUCCESS] Services launched."
echo "API Backend: http://127.0.0.1:9000"
echo "Dashboard:   http://localhost:9090"
echo "========================================================"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "echo 'Shutting down servers...'; kill $API_PID $FRONTEND_PID; exit" SIGINT SIGTERM
wait
