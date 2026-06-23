@echo off
echo ========================================================
echo GridLock R2 - Hackathon Demo Launcher (Instant Inference)
echo ========================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Virtual environment not found. Creating and installing dependencies...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    echo [1/3] Activating venv...
    call venv\Scripts\activate
)

echo [2/4] Generating SHAP Feature Importance Analysis...
python src/evaluation/shap_analysis.py

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] SHAP analysis failed. Dashboard will show empty feature importance chart.
)

echo.
echo [3/4] Starting FastAPI Backend Server on port 9000...
start /min "GridLock R2 API" cmd /c "call venv\Scripts\activate && uvicorn src.api.main:app --host 127.0.0.1 --port 9000"

echo [4/4] Starting Frontend Server on port 9090...
start /min "GridLock R2 Dashboard" cmd /c "python -m http.server 9090 --directory docs"

echo.
echo ========================================================
echo [SUCCESS] Services launched in background windows.
echo API Backend: http://127.0.0.1:9000
echo Dashboard:   http://localhost:9090
echo ========================================================
echo.
echo Launching dashboard in your default browser...
timeout /t 2 >nul
start http://localhost:9090

echo Press any key to shutdown all servers and close this window.
pause >nul

echo Shutting down...
taskkill /FI "WINDOWTITLE eq GridLock R2 API*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq GridLock R2 Dashboard*" /T /F >nul 2>&1
exit
