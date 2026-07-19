@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] No venv. Run python -m venv .venv first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python complete_automation_gui.py
if errorlevel 1 pause
