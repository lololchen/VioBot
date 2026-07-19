@echo off
REM ============================================================
REM  Viobot GUI hub - one-click launcher (D-030)
REM
REM  Double-click: opens ONE browser window with all module tabs
REM  (MelodyExtractor / Sound2Motion / Firmware / AudioFeedback).
REM  First run: creates .venv and installs both packages.
REM
REM  Crash handling lives in gui_hub\launch_all.py (it restarts a
REM  dead server); this bat only restarts the hub itself.
REM
REM  Written goto-style on purpose: parenthesized if-blocks are
REM  fragile in cmd. Keep CRLF line endings (cmd misparses LF).
REM ============================================================
setlocal

REM Work from this script's own directory (handles spaces in path).
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "MARKER=%~dp0.venv\Scripts\motion-planner.exe"

if not exist "%VENV_PY%" goto bootstrap
if exist "%MARKER%" goto run

REM ---- .venv exists but MotionPlanner missing: install just that ----
echo [setup] Installing MotionPlanner into the existing .venv...
"%VENV_PY%" -m pip install -e "%~dp0MotionPlanner[gui]"
if errorlevel 1 goto fail_install
goto run

:bootstrap
echo [setup] No environment found. Creating .venv in this folder...
echo [setup] This is a one-time step and may take a few minutes.

where python >nul 2>nul
if errorlevel 1 goto nopython

python -m venv "%~dp0.venv"
if errorlevel 1 goto fail_venv

echo [setup] Installing packages (melody-extractor + motion-planner + GUIs)...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -e "%~dp0MelodyExtractor[gui,ffmpeg]" -e "%~dp0MotionPlanner[gui]"
if errorlevel 1 goto fail_install
echo [setup] Done.

:run
echo [run] Starting the Viobot GUI hub (all module tabs)...
echo [run] One browser window should open. Close this window to stop everything.

:run_again
"%VENV_PY%" "%~dp0gui_hub\launch_all.py"
if errorlevel 1 goto crashed
goto end

:crashed
echo.
echo [warn] The GUI hub stopped with an error (details above).
echo [warn] Restarting in 3 seconds... Close this window to stop for good.
timeout /t 3 /nobreak >nul
goto run_again

:nopython
echo [error] No Python found on PATH. Install Python 3.10+ and retry.
pause
goto end

:fail_venv
echo [error] Failed to create the virtual environment.
pause
goto end

:fail_install
echo [error] Package installation failed. See messages above.
pause
goto end

:end
endlocal
