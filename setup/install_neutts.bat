@echo off
title ChangeGUI -- Install NeuTTS Client
echo.
echo ============================================================
echo  Installing NeuTTS (Gradio client + supporting libs)
echo ============================================================
echo.
echo  This installs the Python client that connects to a
echo  running NeuTTS Gradio server.
echo.
echo  The NeuTTS SERVER itself has its own requirements in:
echo    VoiceModules/NeuTTS/requirements.txt
echo.
echo  Press any key to continue, or close this window to skip.
echo ============================================================
pause >nul

echo.
echo Installing gradio-client + librosa...
pip install -r requirements_neutts.txt

echo.
echo ============================================================
if %errorlevel% equ 0 (
    echo  NeuTTS client installed successfully.
    echo.
    echo  To use NeuTTS, you also need a running Gradio server.
    echo  See VoiceModules/NeuTTS/ for setup instructions.
) else (
    echo  FAILED. Try running as Administrator.
)
echo ============================================================
echo.
pause
