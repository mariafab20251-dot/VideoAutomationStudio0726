@echo off
title ChangeGUI -- Install Qwen3-TTS
echo.
echo ============================================================
echo  Installing Qwen3-TTS (local AI voice)
echo ============================================================
echo.
echo  This adds Qwen3-TTS support - requires a GPU with CUDA
echo  for reasonable performance.
echo.
echo  This will download PyTorch (several GB) - make sure
echo  you have a stable internet connection.
echo.
echo  Press any key to continue, or close this window to skip.
echo ============================================================
pause >nul

echo.
echo Installing PyTorch + Qwen3-TTS...
pip install -r requirements_qwen.txt

echo.
echo ============================================================
if %errorlevel% equ 0 (
    echo  Qwen3-TTS installed successfully.
) else (
    echo  FAILED. Try running as Administrator.
    echo  If torch fails, you may need to install CUDA first.
)
echo ============================================================
echo.
pause
