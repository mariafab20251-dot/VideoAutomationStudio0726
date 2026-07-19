@echo off
title ChangeGUI — Install Core Dependencies
echo.
echo ============================================================
echo  ChangeGUI — Installing Core Dependencies
echo ============================================================
echo.
echo  This will install the packages needed to run the tool.
echo  Make sure Python is installed and added to PATH first.
echo.
echo  Press any key to continue, or close this window to cancel.
echo ============================================================
pause >nul

echo.
echo [1/2] Upgrading pip ...
python -m pip install --upgrade pip

echo.
echo [2/2] Installing core packages (this may take a few minutes)...
pip install -r requirements_core.txt

echo.
echo ============================================================
if %errorlevel% equ 0 (
    echo  SUCCESS! Core dependencies installed.
    echo.
    echo  You can now run the tool:
    echo    python complete_automation_gui.py
    echo.
    echo  Optional voice modules (run if needed):
    echo    install_kokoro.bat   --- Kokoro TTS (local ONNX voice)
    echo    install_piper.bat    --- Piper TTS (native C++, fastest)
    echo    install_qwen.bat     --- Qwen3-TTS (local AI voice)
    echo    install_neutts.bat   --- NeuTTS (Gradio server voice)
) else (
    echo  SOME PACKAGES FAILED. Check the errors above.
    echo  Try running this batch file as Administrator.
)
echo ============================================================
echo.
pause
