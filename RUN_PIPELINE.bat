@echo off
setlocal
cd /d "%~dp0"
python run_pipeline.py --config config.yaml
if errorlevel 1 (
  echo.
  echo Pipeline failed. Check the message above.
  pause
  exit /b 1
)
echo.
echo Pipeline completed.
pause
