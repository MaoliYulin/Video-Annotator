@echo off
setlocal
cd /d "%~dp0"
set "SOURCE_MODE=browser_upload"
if not exist ".venv\Scripts\python.exe" (
  echo Creating isolated Python environment...
  py -3 -m venv .venv
  if errorlevel 1 (pause & exit /b 1)
)
echo Installing dependencies in .venv...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (pause & exit /b 1)
echo Starting at http://127.0.0.1:8000
start "SCAND Local Annotator" "%~dp0run_server.bat"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000"
echo Keep the server window open. Annotations are saved in storage\annotations.sqlite3
pause
