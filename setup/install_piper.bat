@echo off
title ChangeGUI -- Install Piper TTS
echo.
echo ============================================================
echo  Installing Piper TTS (Lightweight ONNX voice engine)
echo ============================================================
echo.

set PIPER_DIR=%~dp0..\VoiceModules\PiperTTS
set BIN_DIR=%PIPER_DIR%\bin

REM Check if already installed
if exist "%BIN_DIR%\piper.exe" (
    echo [SKIP] Piper TTS already installed (piper.exe found).
    echo.
    pause
    exit /b 0
)

echo.
echo  This downloads the Piper TTS engine + a default voice model.
echo  Total size: ~75 MB (engine ~15 MB + voice ~60 MB)
echo.
echo  Press any key to continue, or close this window to skip.
echo ============================================================
pause >nul

set PIPER_DIR=%~dp0..\VoiceModules\PiperTTS
set VOICES_DIR=%PIPER_DIR%\voices
set BIN_DIR=%PIPER_DIR%\bin

echo.
echo [1/5] Creating directories...
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
if not exist "%VOICES_DIR%" mkdir "%VOICES_DIR%"

rem --- Choose downloader: prefer curl.exe (built into Win10+), fallback to PowerShell ---
where curl.exe >nul 2>&1
if %errorlevel% equ 0 (
    set DL_ENGINE=curl
) else (
    set DL_ENGINE=powershell
)

echo [2/5] Downloading Piper engine (Windows x64)...
if "%DL_ENGINE%"=="curl" (
    curl -L --ssl-no-revoke -o "%TEMP%\piper_windows_amd64.zip" "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
) else (
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = 'tls12'; Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip' -OutFile '%TEMP%\piper_windows_amd64.zip'"
)

echo [3/5] Extracting engine files...
powershell -Command "Expand-Archive -Path '%TEMP%\piper_windows_amd64.zip' -DestinationPath '%TEMP%\piper_extract' -Force"
copy /Y "%TEMP%\piper_extract\piper\piper.exe" "%BIN_DIR%" >nul
copy /Y "%TEMP%\piper_extract\piper\onnxruntime.dll" "%BIN_DIR%" >nul
copy /Y "%TEMP%\piper_extract\piper\espeak-ng.dll" "%BIN_DIR%" >nul
if exist "%TEMP%\piper_extract\piper\espeak-ng-data" (
    robocopy "%TEMP%\piper_extract\piper\espeak-ng-data" "%BIN_DIR%\espeak-ng-data" /E /NFL /NDL /NJH /NJS /NP >nul
)

echo [4/5] Downloading default voice (en_US-ryan-medium ~60 MB)...
if "%DL_ENGINE%"=="curl" (
    curl -L --ssl-no-revoke -o "%VOICES_DIR%\en_US-ryan-medium.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
    curl -L --ssl-no-revoke -o "%VOICES_DIR%\en_US-ryan-medium.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
) else (
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = 'tls12'; Invoke-WebRequest -UseBasicParsing -Uri 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx' -OutFile '%VOICES_DIR%\en_US-ryan-medium.onnx'"
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = 'tls12'; Invoke-WebRequest -UseBasicParsing -Uri 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json' -OutFile '%VOICES_DIR%\en_US-ryan-medium.onnx.json'"
)

echo [5/5] Downloading voice catalog...
if "%DL_ENGINE%"=="curl" (
    curl -L --ssl-no-revoke -o "%PIPER_DIR%\voices.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
) else (
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = 'tls12'; Invoke-WebRequest -UseBasicParsing -Uri 'https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json' -OutFile '%PIPER_DIR%\voices.json'"
)

echo.
echo ============================================================
if exist "%BIN_DIR%\piper.exe" (
    echo  SUCCESS! Piper TTS installed.
    echo  Default voice: en_US-ryan-medium
    echo.
    echo  To add more voices, open the app and use
    echo  the "Download More Voices" button in Piper settings.
) else (
    echo  FAILED: piper.exe not found at %BIN_DIR%\piper.exe
    echo  Try running this batch file as Administrator.
)
echo ============================================================
echo.
pause
