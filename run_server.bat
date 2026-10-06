@echo off
cd /d "%~dp0"
set "SOURCE_MODE=browser_upload"
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
