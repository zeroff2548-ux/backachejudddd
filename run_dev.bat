@echo off
chcp 65001 >nul
rem เปิดแบบมีหน้าต่างคำสั่ง ไว้ดู error (ใช้ START.bat ติดตั้งก่อน 1 ครั้ง)
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo ยังไม่ได้ติดตั้ง กรุณาดับเบิลคลิก START.bat ก่อน
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python posture_shimeji.py
pause
