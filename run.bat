@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if exist "assets\about.nfo" type "assets\about.nfo"
if not exist ".venv\Scripts\python.exe" (
  echo Run installer.bat from your download or repository folder first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
  echo.
  echo The app stopped with an error. Review the message above.
  pause
)
