@echo off
title ChangeGUI — Install Kokoro TTS
echo.
echo ============================================================
echo  Installing Kokoro TTS (local ONNX voice)
echo ============================================================
echo.

set "KOKORO_DIR=%~dp0..\VoiceModules\KokoroTTS"

REM Check if already installed
if exist "%KOKORO_DIR%\kokoro-v0_19.onnx" if exist "%KOKORO_DIR%\voices-v1.0.bin" (
    pip show kokoro-onnx >nul 2>&1
    if not errorlevel 1 (
        echo [SKIP] Kokoro TTS already fully installed. Nothing to do.
        echo.
        pause
        exit /b 0
    )
)

echo.
echo  This adds Kokoro TTS support — voice synthesis runs
echo  locally on your PC using ONNX runtime (no GPU needed).
echo.
echo  Press any key to continue, or close this window to skip.
echo ============================================================
pause >nul

echo.
echo Installing kokoro-onnx...
pip install -r requirements_kokoro.txt

echo.
echo ============================================================
echo  Downloading Kokoro model files (~330 MB, one time)
echo ============================================================
echo.

REM Model files are NOT shipped in the repo (too large). Fetch them here.
set "KOKORO_DIR=%~dp0..\VoiceModules\KokoroTTS"
if not exist "%KOKORO_DIR%" mkdir "%KOKORO_DIR%"

if exist "%KOKORO_DIR%\kokoro-v0_19.onnx" (
    echo  Model already present — skipping download.
) else (
    echo  Downloading kokoro-v0_19.onnx ...
    curl -L -o "%KOKORO_DIR%\kokoro-v0_19.onnx" "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
    echo  Downloading voices-v1.0.bin ...
    curl -L -o "%KOKORO_DIR%\voices-v1.0.bin" "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)

echo.
echo ============================================================
if exist "%KOKORO_DIR%\kokoro-v0_19.onnx" (
    echo  Kokoro TTS installed successfully.
) else (
    echo  FAILED. Check your internet connection, or download the model
    echo  files manually into VoiceModules\KokoroTTS\ :
    echo    - kokoro-v0_19.onnx
    echo    - voices-v1.0.bin
    echo  Source: https://github.com/thewh1teagle/kokoro-onnx/releases
)
echo ============================================================
echo.
pause
