@echo off
setlocal enabledelayedexpansion
title ChangeGUI -- Install Dubbing (CPU / No NVIDIA GPU)
echo.
echo ============================================================
echo  Installing Dubbing tab -- CPU stack (no NVIDIA GPU)
echo ============================================================
echo.
echo  Use this installer on PCs with integrated or AMD shared GPU.
echo  For a dedicated NVIDIA GPU, use setup\install_dubbing.bat instead.
echo.
echo  Press any key to continue, or close this window to skip.
echo ============================================================
pause >nul

REM ---- Locate Python 3.11 -----------------------------------------
set "PY311="
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python311\python.exe"
    "C:\Program Files\Python311\python.exe"
) do (
    if exist %%P (
        if not defined PY311 set "PY311=%%~P"
    )
)
REM Also try py launcher
if not defined PY311 (
    py -3.11 --version >nul 2>&1
    if !errorlevel! equ 0 set "PY311=py -3.11"
)
if not defined PY311 (
    echo   ERROR: Python 3.11 not found. Install it from python.org then retry.
    pause & exit /b 1
)
echo   Python 3.11 : %PY311%

REM ---- Paths -------------------------------------------------------
set "HERE=%~dp0"
set "REPO=%HERE%..\..\"
set "VENV=%HERE%..\..\setup\dub_venv"
set "VPY=%VENV%\Scripts\python.exe"

REM ---- Create venv if needed --------------------------------------
if exist "%VPY%" (
    echo [SKIP] dub_venv already exists.
) else (
    echo [1/6] Creating dub_venv (Python 3.11) ...
    %PY311% -m venv "%VENV%"
    if !errorlevel! neq 0 ( echo   ERROR: venv creation failed. & pause & exit /b 1 )
)

REM ---- Detect offline wheelhouse ----------------------------------
set "WHEELS=%HERE%..\wheels"
if exist "%WHEELS%\" (
    echo   Wheelhouse  : %WHEELS%  ^(OFFLINE install^)
    set "OFF=--no-index --find-links "%WHEELS%""
    set "ISOFF=1"
) else (
    echo   Wheelhouse  : none -- will download from the internet
    set "OFF="
    set "ISOFF=0"
)
echo.

REM ---- Step 2: torch 2.5.1 CPU ------------------------------------
"%VPY%" -c "import torch" >nul 2>&1
if !errorlevel! equ 0 (
    echo [SKIP] torch already installed.
) else (
    echo [2/6] Installing torch 2.5.1 + torchaudio 2.5.1 (CPU) ...
    if "!ISOFF!"=="1" (
        "%VPY%" -m pip install !OFF! torch==2.5.1 torchaudio==2.5.1
    ) else (
        "%VPY%" -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    )
    if !errorlevel! neq 0 ( echo   ERROR: torch install failed. & pause & exit /b 1 )
)

REM ---- Step 3: main requirements ----------------------------------
echo [3/6] Installing dubbing CPU requirements ...
"%VPY%" -m pip install !OFF! -r "%HERE%requirements_dubbing_cpu.txt"
if !errorlevel! neq 0 ( echo   ERROR: requirements install failed. & pause & exit /b 1 )

REM ---- Step 4: speechbrain 0.5.16 (--no-deps) ---------------------
echo [4/6] Pinning speechbrain==0.5.16 (--no-deps) ...
"%VPY%" -m pip install !OFF! --no-deps speechbrain==0.5.16
if !errorlevel! neq 0 ( echo   ERROR: speechbrain install failed. & pause & exit /b 1 )

REM ---- Step 5: huggingface_hub 0.23.5 (--no-deps) -----------------
REM CPU installs may hit HF hub; pin to 0.23.5 to keep use_auth_token support.
echo [5/6] Pinning huggingface_hub==0.23.5 (--no-deps) ...
"%VPY%" -m pip install !OFF! --no-deps huggingface_hub==0.23.5
if !errorlevel! neq 0 ( echo   ERROR: huggingface_hub install failed. & pause & exit /b 1 )

REM ---- Step 6: restore model weights ------------------------------
echo [6/6] Restoring bundled model weights ...

REM -- demucs htdemucs weight -> torch cache
set "DEMUCS_SRC=%HERE%..\models\demucs\955717e8-8726e21a.th"
set "TORCH_CKPT=%USERPROFILE%\.cache\torch\hub\checkpoints"
set "DEMUCS_DST=%TORCH_CKPT%\955717e8-8726e21a.th"
if exist "%DEMUCS_DST%" (
    echo       [SKIP] htdemucs model already in torch cache.
) else if exist "%DEMUCS_SRC%" (
    if not exist "%TORCH_CKPT%" mkdir "%TORCH_CKPT%"
    copy /Y "%DEMUCS_SRC%" "%DEMUCS_DST%" >nul
    echo       htdemucs model restored.
) else (
    echo       NOTE: bundled htdemucs not found -- demucs will download ~80 MB on first use.
)

REM -- pyannote models -> repo root models\pyannote
if exist "%REPO%models\pyannote\speaker-diarization-3.1\config.yaml" (
    echo       [SKIP] pyannote models already at models\pyannote.
) else if exist "%HERE%..\models\pyannote\speaker-diarization-3.1\config.yaml" (
    echo       Copying pyannote models to models\pyannote ...
    xcopy /E /I /Y /Q "%HERE%..\models\pyannote" "%REPO%models\pyannote" >nul
) else (
    echo       NOTE: no bundled pyannote models found.
)

REM -- whisper models -> repo root models\whisper
if exist "%REPO%models\whisper\faster-whisper-medium\model.bin" (
    echo       [SKIP] whisper medium already at models\whisper.
) else if exist "%HERE%..\models\whisper\faster-whisper-medium\model.bin" (
    echo       Copying whisper models to models\whisper ...
    xcopy /E /I /Y /Q "%HERE%..\models\whisper" "%REPO%models\whisper" >nul
) else (
    echo       NOTE: no bundled whisper models found -- will download on first use.
)

REM ---- Verify -----------------------------------------------------
echo.
echo ============================================================
echo  Verifying the CPU dubbing stack ...
"%VPY%" -c "import warnings;warnings.filterwarnings('ignore');import torch,torchaudio,faster_whisper,pyannote.audio,speechbrain,huggingface_hub,demucs,matplotlib;print('  torch        ',torch.__version__);print('  torchaudio   ',torchaudio.__version__);print('  faster-whisper',faster_whisper.__version__);print('  pyannote.audio',pyannote.audio.__version__);print('  speechbrain  ',speechbrain.__version__);print('  hf_hub       ',huggingface_hub.__version__);print('  demucs       ',demucs.__version__)"
if !errorlevel! neq 0 (
    echo.
    echo   VERIFY FAILED -- check the log above.
    pause & exit /b 1
)
echo ------------------------------------------------------------
echo.
echo  DONE. Open the app, Dubbing tab, tick "Multi-speaker dubbing".
echo ============================================================
echo.
pause
endlocal
