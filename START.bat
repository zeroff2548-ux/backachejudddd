@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Posture Shimeji

rem ===== ดับเบิลคลิกไฟล์นี้ไฟล์เดียวจบ =====
rem 1) ถ้ามี PostureShimeji.exe อยู่แล้ว  -> เปิดเลย
rem 2) ถ้าไม่มี -> ติดตั้งครั้งแรกอัตโนมัติ (5-10 นาที ทำครั้งเดียว) แล้วเปิดโปรแกรม

if exist "PostureShimeji.exe" ( start "" "PostureShimeji.exe" & exit /b 0 )
if exist "dist\PostureShimeji.exe" ( start "" "dist\PostureShimeji.exe" & exit /b 0 )
if exist ".venv\Scripts\pythonw.exe" goto :launch

echo ============================================
echo   ติดตั้งครั้งแรก (ทำครั้งเดียว) กรุณารอสักครู่
echo   ต้องต่ออินเทอร์เน็ตเพื่อโหลดไลบรารี
echo ============================================

set PYEXE=
for %%V in (3.11 3.12 3.10) do (
  if not defined PYEXE (
    py -%%V -c "import sys" >nul 2>&1 && set PYEXE=py -%%V
  )
)
if not defined PYEXE goto :nopython

%PYEXE% -m venv .venv || goto :fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || goto :fail
pip install -r requirements.txt || goto :fail

:launch
start "" ".venv\Scripts\pythonw.exe" posture_shimeji.py
exit /b 0

:nopython
echo.
echo [ไม่พบ Python] กรุณาติดตั้ง Python 3.11 ก่อน (ฟรี):
echo   https://www.python.org/downloads/release/python-3119/
echo ตอนติดตั้งให้ติ๊ก "Add python.exe to PATH" แล้วดับเบิลคลิกไฟล์นี้ใหม่
pause
exit /b 1

:fail
echo.
echo [ผิดพลาด] ติดตั้งไม่สำเร็จ ดูข้อความด้านบน แล้วลองดับเบิลคลิกใหม่อีกครั้ง
pause
exit /b 1
