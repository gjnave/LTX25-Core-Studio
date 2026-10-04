@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if exist "assets\about.nfo" type "assets\about.nfo"
if not exist ".venv\Scripts\python.exe" (
  echo Run ..\1-INSTALL-LTX25-Core-Studio.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
  echo.
  echo The app stopped with an error. Review the message above.
  pause
)
