@echo off
echo ========================================================
echo GridLock R2 - Full Pipeline (Train from Scratch)
echo ========================================================
echo.

echo [1/3] Creating Python Virtual Environment...
if not exist venv\Scripts\activate.bat (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed.
        exit /b %errorlevel%
    )
) else (
    echo Virtual environment already exists. Skipping creation.
)

echo [2/3] Activating venv and installing dependencies...
call venv\Scripts\activate
pip install --upgrade pip --no-cache-dir
pip install -r requirements.txt --no-cache-dir
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    exit /b %errorlevel%
)

echo [3/3] Running End-to-End Pipeline (Training from Scratch)...
python -m src.data.pipeline

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline execution failed.
    exit /b %errorlevel%
)

echo.
echo [4/4] Generating SHAP Feature Importance Analysis...
python src/evaluation/shap_analysis.py

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] SHAP analysis failed. Dashboard will show empty feature importance chart.
)

echo.
echo ========================================================
echo [SUCCESS] Pipeline completed successfully.
echo ========================================================
echo.
echo [5/5] Starting servers for demo...
start /min "GridLock R2 API" cmd /c "call venv\Scripts\activate && uvicorn src.api.main:app --host 127.0.0.1 --port 9000"
start /min "GridLock R2 Dashboard" cmd /c "python -m http.server 9090 --directory docs"
echo Launching dashboard in browser...
timeout /t 2 >nul
start http://localhost:9090
pause
