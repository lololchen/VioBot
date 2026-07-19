@echo off
REM ============================================================
REM  Viobot / MelodyExtractor GUI - one-click launcher
REM
REM  Double-click this file to start the Streamlit GUI.
REM  First run: creates a local .venv in this folder and installs
REM  everything it needs. Later runs: launches instantly.
REM
REM  Written goto-style on purpose: parenthesized if-blocks are
REM  fragile in cmd. Keep CRLF line endings (cmd misparses LF).
REM ============================================================
setlocal

REM Work from this script's own directory (handles spaces in path).
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "GUI_EXE=%~dp0.venv\Scripts\melody-extractor-gui.exe"

if exist "%GUI_EXE%" goto run

REM ---- Bootstrap: .venv missing, create and install ----------
echo [setup] No environment found. Creating .venv in this folder...
echo [setup] This is a one-time step and may take a few minutes.

where python >nul 2>nul
if errorlevel 1 goto nopython

python -m venv "%~dp0.venv"
if errorlevel 1 goto fail_venv

echo [setup] Installing packages (melody-extractor + GUI)...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -e "%~dp0MelodyExtractor[gui,ffmpeg]"
if errorlevel 1 goto fail_install
if not exist "%GUI_EXE%" goto fail_install
echo [setup] Done.

:run
echo [run] Starting the MelodyExtractor GUI...
echo [run] A browser tab should open. Close this window to stop the app.
set "EXTRA_ARGS="

:run_again
"%GUI_EXE%" %* %EXTRA_ARGS%
if errorlevel 1 goto crashed
goto end

REM ---- Crash auto-restart: if the server dies (out of memory, bug, ...)
REM the browser shows "Connection error". Restart the server so a browser
REM refresh reconnects. Headless on restarts: don't open yet another tab.
:crashed
echo.
echo [warn] The GUI server stopped with an error (details above).
echo [warn] Restarting in 3 seconds so you can refresh the browser tab...
echo [warn] Close this window (or press Ctrl+C) to stop for good.
timeout /t 3 /nobreak >nul
set "EXTRA_ARGS=--server.headless=true"
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
