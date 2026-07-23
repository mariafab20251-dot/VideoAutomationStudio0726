@echo off
setlocal enabledelayedexpansion
title ChangeGUI -- Build offline wheelhouse for the Dubbing tab

REM ===================================================================
REM  download_dubbing_wheels.bat   (run on a PC WITH internet)
REM  ------------------------------------------------------------------
REM  Downloads every wheel the Dubbing tab needs into  setup\wheels\
REM  so a target PC can install 100%% OFFLINE with install_dubbing.bat.
REM
REM  Workflow:
REM    1) On a machine with a good connection, run THIS script.
REM    2) Copy the whole ChangeGUI folder (incl. setup\wheels\ and
REM       setup\models\) to the offline PC.
REM    3) On the offline PC, run setup\install_dubbing.bat -- it sees
REM       setup\wheels\ and installs without touching the internet.
REM
REM  Wheels are fetched for cp311 / win_amd64 to match the venv the
REM  installer builds (Python 3.11, 64-bit Windows).
REM ===================================================================

set "HERE=%~dp0"
set "WHEELS=%HERE%wheels"

echo.
echo ============================================================
echo   Building offline wheelhouse -> %WHEELS%
echo ============================================================
echo.

REM ---- Locate Python 3.11 (pip target must match the install venv) --
set "PY311="
py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%i in ('py -3.11 -c "import sys;print(sys.executable)"') do set "PY311=%%i"
)
if not defined PY311 if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY311 (
    echo   ERROR: Python 3.11 not found. Install it and re-run.
    pause
    exit /b 1
)
echo   Using Python: !PY311!
if not exist "%WHEELS%" mkdir "%WHEELS%"

REM ---- torch + torchaudio (CUDA 12.1 build) -------------------------
echo.
echo [1/2] Downloading torch 2.5.1+cu121 (large, several GB) ...
"!PY311!" -m pip download torch==2.5.1+cu121 torchaudio==2.5.1+cu121 ^
    --index-url https://download.pytorch.org/whl/cu121 ^
    --only-binary=:all: --python-version 3.11 --platform win_amd64 ^
    -d "%WHEELS%"
if !errorlevel! neq 0 (
    echo   WARNING: CUDA torch download failed. Trying CPU build ...
    "!PY311!" -m pip download torch==2.5.1 torchaudio==2.5.1 ^
        --only-binary=:all: --python-version 3.11 --platform win_amd64 ^
        -d "%WHEELS%"
)

REM ---- everything else --------------------------------------------
echo.
echo [2/2] Downloading faster-whisper + pyannote + demucs + deps ...
"!PY311!" -m pip download -r "%HERE%requirements_dubbing.txt" ^
    --python-version 3.11 --platform win_amd64 -d "%WHEELS%"
if !errorlevel! neq 0 (
    echo   NOTE: some packages have no pure win_amd64 wheel; retrying
    echo         without the platform pin ^(may grab an sdist^) ...
    "!PY311!" -m pip download -r "%HERE%requirements_dubbing.txt" -d "%WHEELS%"
)

REM ---- speechbrain 0.5.16 (installed --no-deps on target) ----------
echo.
echo [2b] Downloading speechbrain==0.5.16 ...
"!PY311!" -m pip download --no-deps speechbrain==0.5.16 ^
    --python-version 3.11 --platform win_amd64 -d "%WHEELS%"
if !errorlevel! neq 0 "!PY311!" -m pip download --no-deps speechbrain==0.5.16 -d "%WHEELS%"

echo.
echo ============================================================
echo   DONE. Wheelhouse ready at:
echo     %WHEELS%
for /f %%c in ('dir /b "%WHEELS%" 2^>nul ^| find /c /v ""') do echo   Files: %%c
echo.
echo   Copy the ChangeGUI folder to the offline PC and run
echo   setup\install_dubbing.bat there.
echo ============================================================
echo.
pause
