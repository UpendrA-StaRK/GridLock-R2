@echo off
echo ========================================================
echo GridLock R2 - Fresh System Setup ^& Execution
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
pip install --upgrade pip
pip install -r requirements.txt
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
echo ========================================================
echo [SUCCESS] Pipeline completed successfully.
echo ========================================================
pause
