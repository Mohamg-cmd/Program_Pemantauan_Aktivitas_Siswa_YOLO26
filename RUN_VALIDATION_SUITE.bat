@echo off
setlocal
cd /d "%~dp0"
python validation\run_validation_suite.py --config validation\config.yaml
if errorlevel 1 (
  echo Validation suite failed. Check the message above.
  exit /b 1
)
echo Validation suite completed.
