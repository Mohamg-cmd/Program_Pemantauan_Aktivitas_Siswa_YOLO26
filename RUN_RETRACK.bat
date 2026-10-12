@echo off
setlocal
cd /d "%~dp0"
echo WARNING: This explicitly reruns YOLO26-M + BoT-SORT/ReID from the original video.
echo Use only when a tracking rerun is scientifically required.
python run_pipeline.py --config config.yaml --force-track
if errorlevel 1 pause
