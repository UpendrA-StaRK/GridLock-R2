@echo off
echo ========================================================
echo GridLock R2 - Inference Only (Uses Existing Checkpoint)
echo ========================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run run_project.bat first to set up the environment.
    exit /b 1
)

echo [1/2] Activating venv...
call venv\Scripts\activate

echo [2/2] Running End-to-End Pipeline (Skipping Training)...
python -m src.data.pipeline --skip-training --ckpt-dir submission_checkpoint

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline execution failed.
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo [SUCCESS] Inference completed successfully.
echo ========================================================
pause
