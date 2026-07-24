@echo off
setlocal enabledelayedexpansion
title ChangeGUI -- Install Dubbing (AUTO GPU detect: NVIDIA / AMD / CPU)

REM ===================================================================
REM  install_dubbing_auto.bat
REM  ------------------------------------------------------------------
REM  ONE installer that AUTO-DETECTS the GPU and picks the right torch
REM  build for the Dubbing tab, with NO manual choice:
REM
REM     1) NVIDIA (dedicated) -> torch CUDA 12.1  (fastest, GPU dubbing)
REM     2) AMD    (dedicated) -> torch-directml   (GPU dubbing on AMD/Windows)
REM     3) neither            -> torch CPU        (works everywhere)
REM
REM  AMD is checked BEFORE falling back to CPU, exactly as intended:
REM  an AMD Radeon PC gets GPU-accelerated dubbing via DirectML, not CPU.
REM
REM  Everything else (faster-whisper, pyannote, demucs, speechbrain,
REM  model-weight restore) is identical across all three paths.
REM  Offline wheelhouse (setup\wheels\) is honoured when present.
REM ===================================================================

set "HERE=%~dp0"
set "REPO=%HERE%.."
set "VENV=%HERE%dub_venv"
set "VPY=%VENV%\Scripts\python.exe"
set "WHEELS=%HERE%wheels"

echo.
echo ============================================================
echo   ChangeGUI  --  Dubbing installer  (auto GPU detect)
echo ============================================================
echo   Target venv : %VENV%
if exist "%WHEELS%\" (
    echo   Wheelhouse  : %WHEELS%  ^(OFFLINE install^)
    set "OFFLINE=1"
) else (
    echo   Wheelhouse  : none found -- will download from the internet
    set "OFFLINE=0"
)
echo ============================================================
echo.

REM ---- 0) DETECT GPU VENDOR -----------------------------------------
REM  GPU=NVIDIA / AMD / CPU. NVIDIA wins if both are present.
echo [0/6] Detecting GPU ...
set "GPU=CPU"

REM  NVIDIA: nvidia-smi ships with the driver; its presence == usable CUDA GPU.
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 set "GPU=NVIDIA"
)

REM  AMD: only look if NVIDIA wasn't found. Query the video controllers and
REM  match AMD / Radeon in the name (skip the CPU's integrated Intel/UHD).
if "!GPU!"=="CPU" (
    for /f "skip=1 delims=" %%G in ('wmic path win32_VideoController get Name 2^>nul') do (
        echo %%G | findstr /i "AMD Radeon RX Vega FirePro" >nul 2>&1
        if !errorlevel! equ 0 set "GPU=AMD"
    )
)

if "!GPU!"=="NVIDIA" echo       Detected: NVIDIA GPU  -^> CUDA 12.1 build ^(GPU dubbing^)
if "!GPU!"=="AMD"    echo       Detected: AMD GPU     -^> DirectML build ^(GPU dubbing^)
if "!GPU!"=="CPU"    echo       Detected: no dedicated GPU -^> CPU build
echo.

REM ---- 1) Locate Python 3.11 ----------------------------------------
echo [1/6] Locating Python 3.11 ...
set "PY311="
py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%i in ('py -3.11 -c "import sys;print(sys.executable)"') do set "PY311=%%i"
)
if not defined PY311 (
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "C:\Python311\python.exe"
        "C:\Program Files\Python311\python.exe"
    ) do if not defined PY311 if exist %%P set "PY311=%%~P"
)
if not defined PY311 (
    echo.
    echo   ERROR: Python 3.11 not found.
    echo   Install it from https://www.python.org/downloads/release/python-3119/
    echo   ^(tick "Add python.exe to PATH"^), then re-run this script.
    echo   3.11 is required -- pyannote 3.x does not support 3.13/3.14.
    echo.
    pause
    exit /b 1
)
echo       Using: !PY311!

REM ---- 2) Create the venv (idempotent) ------------------------------
echo [2/6] Creating venv at %VENV% ...
if exist "%VPY%" (
    echo       [SKIP] venv already exists.
) else (
    "!PY311!" -m venv "%VENV%"
    if not exist "%VPY%" (
        echo   ERROR: venv creation failed.
        pause
        exit /b 1
    )
)
"%VPY%" -m pip install --upgrade pip >nul 2>&1

REM ---- 3) Skip pip if the stack is already complete -----------------
"%VPY%" -c "import torch,faster_whisper,pyannote.audio,demucs" >nul 2>&1
if !errorlevel! equ 0 (
    echo [3/6] Dub stack already installed in venv -- skipping pip.
    goto :models
)

REM ---- 3a) torch (vendor-specific) ----------------------------------
echo [3/6] Installing torch for GPU=!GPU! ...
if "!OFFLINE!"=="1" (
    echo       from wheelhouse ^(offline^) ...
    if "!GPU!"=="NVIDIA" (
        "%VPY%" -m pip install --no-index --find-links "%WHEELS%" torch==2.5.1+cu121 torchaudio==2.5.1+cu121
        if !errorlevel! neq 0 "%VPY%" -m pip install --no-index --find-links "%WHEELS%" torch==2.5.1 torchaudio==2.5.1
    ) else if "!GPU!"=="AMD" (
        "%VPY%" -m pip install --no-index --find-links "%WHEELS%" torch==2.5.1 torchaudio==2.5.1 torch-directml
        if !errorlevel! neq 0 "%VPY%" -m pip install --no-index --find-links "%WHEELS%" torch==2.5.1 torchaudio==2.5.1
    ) else (
        "%VPY%" -m pip install --no-index --find-links "%WHEELS%" torch==2.5.1 torchaudio==2.5.1
    )
) else (
    if "!GPU!"=="NVIDIA" (
        echo       CUDA 12.1 build from the PyTorch index ^(large, several GB^) ...
        "%VPY%" -m pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
        if !errorlevel! neq 0 (
            echo       CUDA build failed -- falling back to CPU build ...
            "%VPY%" -m pip install torch==2.5.1 torchaudio==2.5.1
        )
    ) else if "!GPU!"=="AMD" (
        echo       CPU torch + torch-directml ^(AMD GPU acceleration on Windows^) ...
        "%VPY%" -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
        "%VPY%" -m pip install torch-directml
        if !errorlevel! neq 0 echo       NOTE: torch-directml failed -- dubbing will still run on CPU.
    ) else (
        echo       CPU-only build ...
        "%VPY%" -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    )
)

REM ---- 3b) the rest of the stack -----------------------------------
echo [4/6] Installing faster-whisper + pyannote + demucs ...
if "!OFFLINE!"=="1" (
    "%VPY%" -m pip install --no-index --find-links "%WHEELS%" -r "%HERE%requirements_dubbing.txt"
) else (
    "%VPY%" -m pip install -r "%HERE%requirements_dubbing.txt"
)
if !errorlevel! neq 0 (
    echo   ERROR: dependency install failed. See messages above.
    pause
    exit /b 1
)

REM ---- 4b) speechbrain 0.5.16 (pinned, --no-deps) ------------------
echo       pinning speechbrain==0.5.16 (--no-deps) ...
if "!OFFLINE!"=="1" (
    "%VPY%" -m pip install --no-index --find-links "%WHEELS%" --no-deps speechbrain==0.5.16
) else (
    "%VPY%" -m pip install --no-deps speechbrain==0.5.16
)
if !errorlevel! neq 0 (
    echo   ERROR: speechbrain install failed. See messages above.
    pause
    exit /b 1
)

REM  CPU/AMD paths may hit the HF hub; pin hub 0.23.5 so use_auth_token works.
if not "!GPU!"=="NVIDIA" (
    echo       pinning huggingface_hub==0.23.5 (--no-deps) ...
    if "!OFFLINE!"=="1" (
        "%VPY%" -m pip install --no-index --find-links "%WHEELS%" --no-deps huggingface_hub==0.23.5
    ) else (
        "%VPY%" -m pip install --no-deps huggingface_hub==0.23.5
    )
)

:models
REM ---- 5) Restore bundled model weights ----------------------------
echo [5/6] Restoring bundled model weights ...
call :restore_tree "%HERE%models\whisper"  "%REPO%\models\whisper"  "faster-whisper models"
call :restore_tree "%HERE%models\pyannote" "%REPO%\models\pyannote" "pyannote diarization models"

set "DEMUCS_SRC=%HERE%models\demucs\955717e8-8726e21a.th"
set "TORCH_HOME=%VENV%\torch_cache"
set "DEMUCS_DST=%TORCH_HOME%\hub\checkpoints\955717e8-8726e21a.th"
if exist "%DEMUCS_DST%" (
    echo       [SKIP] htdemucs weight already in venv torch cache.
) else if exist "%DEMUCS_SRC%" (
    if not exist "%TORCH_HOME%\hub\checkpoints" mkdir "%TORCH_HOME%\hub\checkpoints"
    copy /Y "%DEMUCS_SRC%" "%DEMUCS_DST%" >nul
    echo       htdemucs weight restored -- vocal removal works offline.
) else (
    echo       NOTE: bundled htdemucs not found; demucs will download it
    echo             ^(~80 MB^) on first "Keep music ^& SFX" use.
)

REM ---- 6) Verify ----------------------------------------------------
echo [6/6] Verifying the stack ...
echo ------------------------------------------------------------
"%VPY%" -c "import warnings;warnings.filterwarnings('ignore');import torch,faster_whisper,pyannote.audio,demucs,ctranslate2;print('  torch        ',torch.__version__,'(CUDA',torch.cuda.is_available(),')');print('  faster-whisper',faster_whisper.__version__);print('  ctranslate2  ',ctranslate2.__version__);print('  pyannote.audio present, demucs present')"
if !errorlevel! neq 0 (
    echo.
    echo   VERIFY FAILED -- the stack did not import cleanly.
    echo   Try re-running as Administrator, or check the log above.
    pause
    exit /b 1
)
if "!GPU!"=="AMD" (
    "%VPY%" -c "import torch_directml as d;print('  torch-directml : OK, device',d.device())" 2>nul
    if !errorlevel! neq 0 echo   torch-directml : not active -- dubbing runs on CPU on this PC.
)
echo ------------------------------------------------------------

if exist "%REPO%\models\whisper\faster-whisper-medium\model.bin" (
    echo   whisper model   : OK ^(medium bundled^)
) else if exist "%REPO%\models\whisper\faster-whisper-base\model.bin" (
    echo   whisper model   : OK ^(base bundled; medium missing^)
) else (
    echo   whisper model   : NOT bundled -- will download on first dub.
)
if exist "%REPO%\models\pyannote\speaker-diarization-3.1\config.yaml" (
    echo   pyannote bundle : OK ^(offline diarization ready^)
) else (
    echo   pyannote bundle : MISSING -- multi-speaker needs an HF token,
    echo                     or copy setup\models\pyannote from the source PC.
)

echo.
echo ============================================================
echo   DONE. GPU path used: !GPU!
echo   The Dubbing tab is ready. Open the app -^> Dubbing tab.
echo   No further configuration needed -- the app auto-detects
echo   this venv at setup\dub_venv.
echo ============================================================
echo.
pause
exit /b 0

REM ===================================================================
REM  :restore_tree  <src>  <dst>  <label>
REM ===================================================================
:restore_tree
set "SRC=%~1"
set "DST=%~2"
set "LBL=%~3"
if not exist "%SRC%\" (
    echo       [SKIP] %LBL%: no bundle at %SRC%
    goto :eof
)
if exist "%DST%\" (
    dir /a /b "%DST%" 2>nul | findstr "." >nul
    if !errorlevel! equ 0 (
        echo       [SKIP] %LBL%: already present at %DST%
        goto :eof
    )
)
echo       Copying %LBL% ...
xcopy /E /I /Y /Q "%SRC%" "%DST%" >nul
echo       %LBL%: restored to %DST%
goto :eof
