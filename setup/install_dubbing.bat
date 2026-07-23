@echo off
setlocal enabledelayedexpansion
title ChangeGUI -- Install Dubbing (self-contained, offline-capable)

REM ===================================================================
REM  install_dubbing.bat
REM  ------------------------------------------------------------------
REM  ONE batch file that sets up EVERYTHING the Dubbing tab needs on a
REM  fresh PC, with NO manual steps:
REM     * a dedicated Python 3.11 venv at  setup\dub_venv
REM     * torch (CUDA 12.1, CPU fallback) + faster-whisper + pyannote
REM       + demucs, pinned to the exact working versions
REM     * restores the bundled model weights (whisper / pyannote /
REM       demucs) into the folders the engine looks in
REM  If a wheelhouse (setup\wheels\) is present it installs 100%% OFFLINE.
REM  Otherwise it downloads from PyPI + the PyTorch index.
REM
REM  The main app runs on Python 3.13; this venv is separate on purpose
REM  (pyannote 3.x needs the torch 2.5 / cuDNN-9 stack). dubbing_engine
REM  auto-detects setup\dub_venv\Scripts\python.exe, so nothing else to
REM  configure after this finishes.
REM ===================================================================

set "HERE=%~dp0"
set "REPO=%HERE%.."
set "VENV=%HERE%dub_venv"
set "VPY=%VENV%\Scripts\python.exe"
set "WHEELS=%HERE%wheels"

echo.
echo ============================================================
echo   ChangeGUI  --  Dubbing tab installer  (self-contained)
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

REM ---- 1) Locate Python 3.11 ----------------------------------------
echo [1/6] Locating Python 3.11 ...
set "PY311="
py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%i in ('py -3.11 -c "import sys;print(sys.executable)"') do set "PY311=%%i"
)
if not defined PY311 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    )
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

REM ---- 3) Skip if the stack is already complete ---------------------
"%VPY%" -c "import torch,faster_whisper,pyannote.audio,demucs" >nul 2>&1
if !errorlevel! equ 0 (
    echo [3/6] Dub stack already installed in venv -- skipping pip.
    goto :models
)

REM ---- 3a) torch (GPU first, CPU fallback) --------------------------
echo [3/6] Installing torch 2.5.1 ...
if "!OFFLINE!"=="1" (
    echo       from wheelhouse ^(offline^) ...
    "%VPY%" -m pip install --no-index --find-links "%WHEELS%" torch==2.5.1 torchaudio==2.5.1
    if !errorlevel! neq 0 "%VPY%" -m pip install --no-index --find-links "%WHEELS%" torch==2.5.1+cu121 torchaudio==2.5.1+cu121
) else (
    echo       CUDA 12.1 build from the PyTorch index ^(large, several GB^) ...
    "%VPY%" -m pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    if !errorlevel! neq 0 (
        echo       CUDA build failed -- falling back to the CPU-only build ...
        "%VPY%" -m pip install torch==2.5.1 torchaudio==2.5.1
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
REM  pyannote 3.3.2 metadata demands speechbrain>=1.0.0, but its runtime
REM  works with 0.5.16 (verified). Install it separately with --no-deps so
REM  the resolver above doesn't reject the combo. See requirements_dubbing.txt.
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

:models
REM ---- 5) Restore bundled model weights ----------------------------
echo [5/6] Restoring bundled model weights ...

REM  5a) whisper + pyannote live at the REPO ROOT (models\), where
REM      _whisper_word_timestamps.py looks for them. Copy from the
REM      bundle in setup\models\ if the repo-root copy is missing.
call :restore_tree "%HERE%models\whisper"  "%REPO%\models\whisper"  "faster-whisper models"
call :restore_tree "%HERE%models\pyannote" "%REPO%\models\pyannote" "pyannote diarization models"

REM  5b) demucs htdemucs weight -> the venv's torch hub checkpoints
set "DEMUCS_SRC=%HERE%models\demucs\955717e8-8726e21a.th"
set "TORCH_CKPT=%VENV%\Lib\site-packages\..\..\..\torch_cache\hub\checkpoints"
REM  demucs honours TORCH_HOME; point it at a cache INSIDE the venv so
REM  it's portable and username-independent. The engine sets the same.
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
"%VPY%" -c "import torch,faster_whisper,pyannote.audio,demucs,ctranslate2;print('  torch        ',torch.__version__,'(CUDA',torch.cuda.is_available(),')');print('  faster-whisper',faster_whisper.__version__);print('  ctranslate2  ',ctranslate2.__version__);print('  pyannote.audio present, demucs present')"
if !errorlevel! neq 0 (
    echo.
    echo   VERIFY FAILED -- the stack did not import cleanly.
    echo   Try re-running as Administrator, or check the log above.
    pause
    exit /b 1
)
echo ------------------------------------------------------------

REM  model-presence check
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
echo   DONE. The Dubbing tab is ready.
echo   Open the app -^> Dubbing tab. For multi-speaker, tick
echo   "Multi-speaker dubbing" then "Detect Speakers".
echo   No further configuration needed -- the app auto-detects
echo   this venv at setup\dub_venv.
echo ============================================================
echo.
pause
exit /b 0

REM ===================================================================
REM  :restore_tree  <src>  <dst>  <label>
REM  Copies src -> dst only when dst is missing/empty. Used for the
REM  large model folders so a re-run doesn't re-copy gigabytes.
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
