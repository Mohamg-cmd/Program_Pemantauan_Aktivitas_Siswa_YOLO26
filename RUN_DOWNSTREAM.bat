@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo YOLO26 - DOWNSTREAM ANALYSIS MODE
echo Uses existing trajectory_botsort.jsonl
echo No detector/tracker rerun.
echo ================================================
python run_pipeline.py --config config.yaml --prefer-trajectory
if errorlevel 1 (
  echo.
  echo Pipeline FAILED. Read the error above.
  pause
  exit /b 1
)
echo.
echo DONE. Results are in: output\
pause
