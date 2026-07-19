@echo off
cd /d "%~dp0.."
echo ============================================
echo  Video Automation Studio — New PC Setup
echo ============================================
echo.
echo This will set up a fresh Python virtual environment
echo and install dependencies for this tool.
echo.
echo Step 1/4: Virtual environment
echo ------------------------------
if exist ".venv" (
    echo [RENAME] Old .venv found — renaming to .venv_old
    ren ".venv" ".venv_old"
    if errorlevel 1 (
        echo [ERROR] Could not rename .venv. Close any programs using it and retry.
        pause
        exit /b 1
    )
)
echo [CREATE] Creating new virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create venv. Is Python installed?
    pause
    exit /b 1
)
echo [OK] Virtual environment created.

echo.
echo Step 2/4: Install core dependencies
echo ------------------------------------
call .venv\Scripts\activate.bat
pip install -r setup\requirements_core.txt
if errorlevel 1 (
    echo [WARN] Some core dependencies may have failed — check above.
) else (
    echo [OK] Core dependencies installed.
)

echo.
echo Step 3/4: TTS Engines (optional)
echo ---------------------------------
echo Which TTS engines do you want on this machine?
echo.
echo  [1] Kokoro  (fast, ~500 MB)
echo  [2] NeuTTS  (medium, ~2 GB)
echo  [3] Qwen3   (best quality, ~15 GB, needs NVIDIA GPU)
echo  [4] ALL of the above
echo  [0] Skip — install later
echo.
set /p TTS_CHOICE="Enter choice (0-4): "

if "%TTS_CHOICE%"=="1" (
    call setup\install_kokoro.bat
) else if "%TTS_CHOICE%"=="2" (
    call setup\install_neutts.bat
) else if "%TTS_CHOICE%"=="3" (
    call setup\install_qwen.bat
) else if "%TTS_CHOICE%"=="4" (
    call setup\install_kokoro.bat
    call setup\install_neutts.bat
    call setup\install_qwen.bat
) else if "%TTS_CHOICE%"=="0" (
    echo [SKIP] No TTS engines installed. You can run setup\install_*.bat later.
) else (
    echo [SKIP] Invalid choice. Install manually via setup\install_*.bat later.
)

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo Launch the tool with:    run.bat
echo.
pause
