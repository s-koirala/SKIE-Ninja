@echo off
REM SKIE_Ninja Weekly ONNX Model Retraining Script
REM Schedule this with Windows Task Scheduler to run every Sunday at 6:00 PM
REM This ensures fresh models are ready before Monday market open

echo ============================================================
echo SKIE_Ninja Weekly ONNX Model Retraining
echo Started: %date% %time%
echo ============================================================

REM Set project directory
set PROJECT_DIR=C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja
set LOG_FILE=%PROJECT_DIR%\logs\retrain_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log

REM Create logs directory if it doesn't exist
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Run retraining script with copy to NinjaTrader
echo Running retraining script...
python src/python/retrain_onnx_models.py --copy-to-ninjatrader >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Models retrained successfully
    echo Check %LOG_FILE% for details
) else (
    echo ERROR: Retraining failed with error code %ERRORLEVEL%
    echo Check %LOG_FILE% for details
)

echo ============================================================
echo Completed: %date% %time%
echo ============================================================
