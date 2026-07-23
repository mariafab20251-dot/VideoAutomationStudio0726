@echo off
title ChangeGUI -- Install Speaker Diarization
echo.
echo ============================================================
echo  Installing Speaker Diarization (multi-speaker dubbing)
echo ============================================================
echo.
echo  This adds pyannote.audio for detecting who speaks when, so
echo  the Dubbing tab can voice each speaker with its own voice.
echo  A CUDA GPU is strongly recommended (CPU works but is slow).
echo.
echo  IMPORTANT: this does NOT install whisperx on purpose - its
echo  cuDNN-8 pin breaks the GPU stack. Word timestamps use
echo  faster-whisper, which the core install already provides.
echo.
echo  You ALSO need the model weights copied from the source PC:
echo    models\pyannote\   (~32 MB, required for offline diarization)
echo    models\whisper\    (faster-whisper weights, if not present)
echo  See DIARIZATION_SETUP_GUIDE.txt for the copy steps.
echo.
echo  Press any key to continue, or close this window to skip.
echo ============================================================
pause >nul

REM ---- Skip if pyannote.audio already importable -------------------
python -c "import pyannote.audio" >nul 2>&1
if %errorlevel% equ 0 (
    echo [SKIP] pyannote.audio already installed.
    goto :verify_models
)

REM ---- Step 1: GPU torch (CUDA 12.1) -------------------------------
REM Skip if a CUDA-enabled torch is already present.
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    echo [SKIP] CUDA-enabled torch already installed.
) else (
    echo [1/2] Installing CUDA torch 2.5.1+cu121 ...
    echo       (large download, several GB - stable connection needed)
    pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
)

REM ---- Step 2: diarization packages --------------------------------
echo [2/3] Installing pyannote.audio + pinned ASR stack ...
pip install -r requirements_diarize.txt

REM ---- Step 3: Demucs (optional voice removal for dubbing) ---------
REM Powers the "Keep original music & sound effects (remove voices)"
REM toggle. Uses the same torch stack, so it's installed here too.
python -c "import demucs" >nul 2>&1
if %errorlevel% equ 0 (
    echo [SKIP] demucs already installed.
) else (
    echo [3/3] Installing Demucs for vocal removal ...
    pip install demucs
)

REM ---- Restore bundled htdemucs weights into the torch hub cache --
REM The htdemucs model (~80 MB, one .th file) ships in setup\models\demucs
REM so the new PC never has to download it. Demucs loads it from the torch
REM hub checkpoints dir; copy it there. If it's already present, skip.
set DEMUCS_SRC=%~dp0models\demucs\955717e8-8726e21a.th
set TORCH_CKPT=%USERPROFILE%\.cache\torch\hub\checkpoints
set DEMUCS_DST=%TORCH_CKPT%\955717e8-8726e21a.th
if exist "%DEMUCS_DST%" (
    echo [SKIP] htdemucs model already in torch cache.
) else (
    if exist "%DEMUCS_SRC%" (
        echo  Restoring bundled htdemucs model into torch cache ...
        if not exist "%TORCH_CKPT%" mkdir "%TORCH_CKPT%"
        copy /Y "%DEMUCS_SRC%" "%DEMUCS_DST%" >nul
        echo  htdemucs model restored - vocal removal works offline.
    ) else (
        echo  NOTE: bundled htdemucs not found at setup\models\demucs.
        echo  Demucs will download it (~80 MB) on first vocal removal instead.
    )
)

:verify_models
echo.
echo ============================================================
python -c "import pyannote.audio" >nul 2>&1
if %errorlevel% neq 0 (
    echo  FAILED. pyannote.audio not importable.
    echo  Try running as Administrator, or see
    echo  DIARIZATION_SETUP_GUIDE.txt / MULTISPEAKER_DUBBING_PLAN.md.
    echo ============================================================
    echo.
    pause
    exit /b 1
)
echo  Packages OK.

REM ---- Warn if local model bundle is missing -----------------------
set BUNDLE=%~dp0..\models\pyannote\speaker-diarization-3.1\config.yaml
if exist "%BUNDLE%" (
    echo  Local pyannote bundle found - offline diarization ready.
) else (
    echo  WARNING: models\pyannote\ not found.
    echo  Copy it from the source PC (see DIARIZATION_SETUP_GUIDE.txt),
    echo  or set an HF Token in the Dubbing tab for online fallback.
)
echo.
echo  Next: open the app, Dubbing tab, tick "Multi-speaker dubbing",
echo  then click "Detect Speakers".
echo ============================================================
echo.
pause
