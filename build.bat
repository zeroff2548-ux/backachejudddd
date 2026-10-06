@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   Build PostureShimeji.exe  (single file)
echo ============================================

rem --- find a compatible Python (mediapipe 0.10.14 supports 3.9 - 3.12) ---
set PYEXE=
for %%V in (3.11 3.12 3.10) do (
  if not defined PYEXE (
    py -%%V -c "import sys" >nul 2>&1 && set PYEXE=py -%%V
  )
)
if not defined PYEXE (
  echo.
  echo [ERROR] Python 3.10 / 3.11 / 3.12 not found.
  echo Please install Python 3.11 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" / "py launcher", then run this file again.
  pause
  exit /b 1
)
echo Using: %PYEXE%

%PYEXE% -m venv .venv || goto :fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || goto :fail
pip install -r requirements.txt || goto :fail
pip install pyinstaller || goto :fail

set ICONARG=
python posture_shimeji.py --make-icon icon.ico
if exist icon.ico set ICONARG=--icon icon.ico

pyinstaller --noconfirm --clean --onefile --windowed --name PostureShimeji %ICONARG% --collect-all mediapipe posture_shimeji.py || goto :fail

echo.
echo Running self-test on the built exe ...
if exist "%APPDATA%\PostureShimeji\selftest.txt" del "%APPDATA%\PostureShimeji\selftest.txt"
start /wait "" "dist\PostureShimeji.exe" --selftest
if exist "%APPDATA%\PostureShimeji\selftest.txt" (
  type "%APPDATA%\PostureShimeji\selftest.txt"
  echo.
)

echo.
echo ============================================
echo   DONE ->  dist\PostureShimeji.exe
echo   Double-click it to run. (first start may take 10-30 sec)
echo ============================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Build failed. See messages above.
pause
exit /b 1
