@echo off
setlocal enabledelayedexpansion
title ChangeGUI -- Copy gitignored files from a source PC

REM ===================================================================
REM  copy_from_source_pc.bat
REM  ------------------------------------------------------------------
REM  A fresh `git clone` is MISSING the gitignored files needed to run:
REM  API-key settings, Google credentials, and the 1.6 GB model bundle.
REM  This script copies them from a SOURCE folder (your old PC's
REM  ChangeGUI folder, reached over a network share, USB drive, or an
REM  already-mounted path) into THIS clone.
REM
REM  Usage:
REM     setup\copy_from_source_pc.bat  "\\OLDPC\share\ChangeGUI"
REM     setup\copy_from_source_pc.bat  "E:\Backup\ChangeGUI"
REM
REM  If you omit the path it will prompt for it.
REM ===================================================================

set "HERE=%~dp0"
set "REPO=%HERE%.."
pushd "%REPO%"
set "REPO=%CD%"
popd

echo.
echo ============================================================
echo   ChangeGUI  --  copy gitignored files from source PC
echo ============================================================
echo   This clone : %REPO%
echo.

REM ---- Resolve SOURCE path -------------------------------------------
set "SRC=%~1"
if "%SRC%"=="" (
    echo   Enter the path to your SOURCE ChangeGUI folder.
    echo   Examples:  \\OLDPC\Users\me\ChangeGUI    or    E:\Backup\ChangeGUI
    echo.
    set /p SRC="Source folder: "
)
if "%SRC%"=="" ( echo   ERROR: no source path given. & pause & exit /b 1 )

REM strip surrounding quotes if the user typed them
set "SRC=%SRC:"=%"

if not exist "%SRC%\" (
    echo   ERROR: source folder not found: %SRC%
    pause
    exit /b 1
)
if not exist "%SRC%\complete_automation_gui.py" (
    echo   WARNING: %SRC% doesn't look like a ChangeGUI folder
    echo   ^(complete_automation_gui.py not found^). Continue anyway?
    set /p GO="Type Y to proceed: "
    if /i not "!GO!"=="Y" ( echo   Aborted. & pause & exit /b 1 )
)
echo   Source     : %SRC%
echo ============================================================
echo.

REM ---- 1) Small settings + credentials -------------------------------
echo [1/3] Settings + credentials ...
call :copy_file "overlay_settings.json"     "MAIN settings + API keys"
call :copy_file "automation_settings.json"  "dashboard settings"
call :copy_file "processing_paths.json"     "saved folder paths"
call :copy_file ".env"                       "extra env vars (optional)"
if exist "%SRC%\google_credentials\" (
    echo       copying google_credentials\ ...
    xcopy /E /I /Y /Q "%SRC%\google_credentials" "%REPO%\google_credentials" >nul
    echo       google_credentials\ ................. copied
) else (
    echo       [SKIP] google_credentials\ not at source
)

REM ---- 2) Whisper + pyannote model bundle ----------------------------
echo.
echo [2/3] Model bundle ^(setup\models\whisper + pyannote, ~1.6 GB^) ...
call :copy_tree "setup\models\whisper"  "faster-whisper models (1.6 GB)"
call :copy_tree "setup\models\pyannote" "pyannote diarization models (31 MB)"
REM demucs weight already ships in the git clone; copy only if somehow missing
if not exist "%REPO%\setup\models\demucs\955717e8-8726e21a.th" (
    call :copy_tree "setup\models\demucs" "demucs htdemucs weight (81 MB)"
) else (
    echo       [SKIP] demucs weight already present ^(from git clone^)
)

REM ---- 3) Optional: TTS VoiceModules ---------------------------------
echo.
echo [3/3] TTS engine data ^(VoiceModules\, optional, can be large^) ...
echo       These can instead be re-downloaded by setup\install_*.bat.
set /p VM="Copy VoiceModules from source too? (y/N): "
if /i "%VM%"=="Y" (
    if exist "%SRC%\VoiceModules\" (
        echo       copying VoiceModules\ -- this may take a while ...
        xcopy /E /I /Y /Q "%SRC%\VoiceModules" "%REPO%\VoiceModules" >nul
        echo       VoiceModules\ .................... copied
    ) else (
        echo       [SKIP] VoiceModules\ not at source
    )
) else (
    echo       [SKIP] VoiceModules -- install per-engine via setup\install_*.bat
)

echo.
echo ============================================================
echo   DONE. Now run:  setup\setup_new_pc.bat
echo   ^(builds the venvs and installs dependencies^)
echo ============================================================
echo.
pause
exit /b 0

REM ===================================================================
REM  :copy_file  <relpath>  <label>
REM ===================================================================
:copy_file
set "REL=%~1"
set "LBL=%~2"
if exist "%SRC%\%REL%" (
    copy /Y "%SRC%\%REL%" "%REPO%\%REL%" >nul
    echo       %REL% ... %LBL% -- copied
) else (
    echo       [SKIP] %REL% not at source ^(%LBL%^)
)
goto :eof

REM ===================================================================
REM  :copy_tree  <relpath>  <label>   (skips if dest already populated)
REM ===================================================================
:copy_tree
set "REL=%~1"
set "LBL=%~2"
if not exist "%SRC%\%REL%\" (
    echo       [SKIP] %REL% not at source ^(%LBL%^)
    goto :eof
)
if exist "%REPO%\%REL%\" (
    dir /a /b "%REPO%\%REL%" 2>nul | findstr "." >nul
    if !errorlevel! equ 0 (
        echo       [SKIP] %REL% already present here ^(%LBL%^)
        goto :eof
    )
)
echo       copying %REL% -- %LBL% ...
xcopy /E /I /Y /Q "%SRC%\%REL%" "%REPO%\%REL%" >nul
echo       %REL% ... restored
goto :eof
