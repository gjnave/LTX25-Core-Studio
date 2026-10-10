@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Get Going Fast - GGF Spokesman Installer
cd /d "%~dp0"
if exist "disclaimer.md" type "disclaimer.md"
echo.
echo ================================================================
echo   GET GOING FAST - GGF SPOKESMAN INSTALLER
echo ================================================================
echo.
set "APP=%~dp0LTX25-Core-Studio"
if exist "%~dp0app.py" set "APP=%~dp0."
set "MODELS=%APP%\models"
set "CACHE=%~dp0installer-cache"
set "BACKUPS=%~dp0app-backups"
set "SOURCE="
set "SOURCE_NAME="
set "UV=%CACHE%\uv\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%CACHE%\python"
set "UV_PYTHON_DOWNLOADS=automatic"
set "VIRTUAL_ENV="
if /I "%~1"=="/update" if exist "%APP%\update_app.py" if exist "%APP%\.venv\Scripts\python.exe" (
  "%APP%\.venv\Scripts\python.exe" "%APP%\update_app.py"
  exit /b !errorlevel!
)
if not defined LTX_CODEBERG_ZIP set "LTX_CODEBERG_ZIP=https://codeberg.org/Cognibuild/LTX25-Core-Studio/archive/main.zip"
if not defined LTX_GITHUB_ZIP set "LTX_GITHUB_ZIP=https://github.com/gjnave/LTX25-Core-Studio/archive/refs/heads/main.zip"
if not defined LTX_DRIVE_ZIP set "LTX_DRIVE_ZIP=https://drive.google.com/uc?export=download&id=1pbgolgOFWNEP_g7r5kcA7_bl08-9GmJL"
if not exist "%CACHE%" mkdir "%CACHE%"
if not exist "%BACKUPS%" mkdir "%BACKUPS%"
if not exist "%APP%" mkdir "%APP%"
if not exist "%CACHE%" goto :FAILED
if not exist "%BACKUPS%" goto :FAILED
if not exist "%APP%" goto :FAILED

if /I not "%~1"=="/update" if exist "%APP%\app.py" if exist "%APP%\core_worker.py" if exist "%APP%\VERSION" if exist "%APP%\vendor\comfy_core\nodes.py" (
  echo Using the verified app files bundled beside this installer.
  set "SOURCE=%APP%"
  set "SOURCE_NAME=Bundled local copy"
)
if not defined SOURCE (
  echo Checking Codeberg for app source...
  call :TRY_SOURCE "Codeberg" "%LTX_CODEBERG_ZIP%"
)
if not defined SOURCE (
  echo Checking GitHub for app source...
  call :TRY_SOURCE "GitHub" "%LTX_GITHUB_ZIP%"
)
if not defined SOURCE (
  echo Checking Google Drive for app source...
  call :TRY_SOURCE "Google Drive" "%LTX_DRIVE_ZIP%"
)
if not defined SOURCE if exist "%APP%\app.py" if exist "%APP%\VERSION" if exist "%APP%\vendor\comfy_core\nodes.py" (
  echo All online mirrors failed. Reusing the app files already extracted beside this installer.
  set "SOURCE=%APP%"
  set "SOURCE_NAME=Bundled local copy"
)
if not defined SOURCE goto :SOURCE_FAILED
if /I "%~1"=="/update" if /I not "!SOURCE!"=="%APP%" if exist "%APP%\VERSION" if exist "%APP%\.venv\Scripts\python.exe" (
  "%APP%\.venv\Scripts\python.exe" -c "from pathlib import Path; import sys; current=Path(r'%APP%\VERSION').read_text().strip(); candidate=Path(r'!SOURCE!\VERSION').read_text().strip(); sys.exit(0 if tuple(map(int,candidate.split('.'))) > tuple(map(int,current.split('.'))) else 1)" >nul 2>&1
  if errorlevel 1 (
    echo Installed app is the same version or newer than the available mirror. Keeping its files.
    set "SOURCE=%APP%"
    set "SOURCE_NAME=Installed local copy"
  )
)
echo Verified source from !SOURCE_NAME!: !SOURCE!

if /I not "!SOURCE!"=="%APP%" if exist "%APP%\app.py" (
  set "BACKUP=%BACKUPS%\app-%RANDOM%-%RANDOM%"
  mkdir "!BACKUP!"
  if errorlevel 1 goto :FAILED
  echo Backing up existing app code to !BACKUP!...
  robocopy "%APP%" "!BACKUP!" /E /R:1 /W:1 /XD .venv models outputs .gradio __pycache__ .git /XF network_settings.json *.pyc >nul
  if errorlevel 8 goto :FAILED
)
if /I not "!SOURCE!"=="%APP%" (
  echo Installing verified app source without removing user files...
  robocopy "!SOURCE!" "%APP%" /E /R:1 /W:1 /XD .venv models outputs .gradio __pycache__ .git /XF network_settings.json *.pyc >nul
  if errorlevel 8 goto :FAILED
)
if /I "%~1"=="/source-only" (
  echo Source-only smoke test complete. Models and Python were not downloaded.
  exit /b 0
)

echo Preparing portable uv and the app's Python runtime...
if not exist "%CACHE%\uv" mkdir "%CACHE%\uv"
if errorlevel 1 goto :FAILED
curl.exe --fail --location --retry 3 --output "%CACHE%\uv-0.12.24.zip" "https://github.com/astral-sh/uv/releases/download/0.12.24/uv-x86_64-pc-windows-msvc.zip"
if errorlevel 1 goto :FAILED
certutil -hashfile "%CACHE%\uv-0.12.24.zip" SHA256 | findstr /I /C:"7c38608c8a18ee137d748a1773053b07ec8f3a30fab49aebaa6f4e4efeceb019" >nul
if errorlevel 1 goto :FAILED
tar.exe -xf "%CACHE%\uv-0.12.24.zip" -C "%CACHE%\uv"
if errorlevel 1 goto :FAILED
if not exist "%APP%\.venv\Scripts\python.exe" (
  if exist "%APP%\.venv" (
    echo ERROR: Existing environment is incomplete. Extract into a new folder.
    goto :FAILED
  )
  echo Downloading managed Python 3.10 and creating the private environment...
  "%UV%" venv --managed-python --python 3.10 --seed "%APP%\.venv"
  if errorlevel 1 goto :FAILED
)
set "PY=%APP%\.venv\Scripts\python.exe"
set "HF=%APP%\.venv\Scripts\hf.exe"
"%PY%" -c "import sys;assert (3,10) <= sys.version_info[:2] <= (3,11) and sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 (
  echo ERROR: The existing private environment is not 64-bit Python 3.10 or 3.11.
  echo Move that environment aside manually before rerunning this installer.
  pause
  exit /b 1
)

echo Installing CUDA PyTorch and app requirements...
"%UV%" pip install --python "%PY%" --upgrade pip setuptools wheel
if errorlevel 1 goto :FAILED
"%PY%" -c "import torch;assert torch.__version__.startswith('2.10.0') and torch.version.cuda=='13.0'" >nul 2>&1
if errorlevel 1 (
  "%UV%" pip install --python "%PY%" torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
  if errorlevel 1 goto :FAILED
)
"%UV%" pip install --python "%PY%" -r "%APP%\requirements.txt"
if errorlevel 1 goto :FAILED
"%UV%" pip check --python "%PY%"
if errorlevel 1 goto :FAILED
"%PY%" -c "import sys;sys.path.insert(0,sys.argv[1]);import app;app.build_demo();print('Spokesman UI and dependency checks passed.')" "%APP%"
if errorlevel 1 goto :FAILED
if /I "%~1"=="/dependencies-only" exit /b 0

if /I "%~1"=="/update" (
  echo Source and dependencies updated. Start run.bat.
  pause
  exit /b 0
)
echo.
echo Downloading missing LTX 2.5 weights from the original publisher...
echo Accept the model terms first: https://huggingface.co/Lightricks/LTX-2.5
echo If access is denied, run "%HF%" auth login, then rerun this BAT.
echo.
call :DOWNLOAD "diffusion_models\ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
if errorlevel 1 goto :MODEL_FAILED
call :DOWNLOAD "text_encoders\gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
if errorlevel 1 goto :MODEL_FAILED
call :DOWNLOAD "vae\ltx-2.5-video-vae-bf16.safetensors"
if errorlevel 1 goto :MODEL_FAILED
call :DOWNLOAD "vae\ltx-2.5-audio-vae-bf16.safetensors"
if errorlevel 1 goto :MODEL_FAILED
call :DOWNLOAD "latent_upscale_models\ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
if errorlevel 1 goto :MODEL_FAILED

"%PY%" -c "import sys;sys.path.insert(0,r'%APP%');from model_config import missing_files;from pathlib import Path;missing=missing_files(Path(r'%MODELS%'));print('Missing or wrong-size model files:',missing);sys.exit(bool(missing))"
if errorlevel 1 goto :MODEL_FAILED
"%PY%" -c "import torch,gradio;print('CUDA available:',torch.cuda.is_available());print('Gradio:',gradio.__version__);assert torch.cuda.is_available()"
if errorlevel 1 goto :FAILED
echo.
echo Ready. Double-click run.bat to start GGF Spokesman.
pause
exit /b 0

:TRY_SOURCE
set "ARCHIVE=%CACHE%\source-%RANDOM%-%RANDOM%.zip"
set "STAGE=%CACHE%\stage-%RANDOM%-%RANDOM%"
mkdir "!STAGE!"
if errorlevel 1 exit /b 1
curl.exe --fail --location --retry 2 --connect-timeout 15 --max-time 180 --output "!ARCHIVE!" "%~2"
if errorlevel 1 exit /b 1
tar.exe -tf "!ARCHIVE!" >nul 2>&1
if errorlevel 1 exit /b 1
tar.exe -xf "!ARCHIVE!" -C "!STAGE!"
if errorlevel 1 exit /b 1
for /d %%D in ("!STAGE!\*") do (
  if exist "%%~fD\app.py" if exist "%%~fD\core_worker.py" if exist "%%~fD\requirements.txt" if exist "%%~fD\VERSION" if exist "%%~fD\vendor\comfy_core\nodes.py" (
    set "SOURCE=%%~fD"
    set "SOURCE_NAME=%~1"
  )
)
if not defined SOURCE exit /b 1
exit /b 0

:DOWNLOAD
if exist "%MODELS%\%~1" (
  echo Reusing %~1
  exit /b 0
)
"%HF%" download Lightricks/LTX-2.5 "%~1" --local-dir "%MODELS%"
exit /b %errorlevel%

:SOURCE_FAILED
echo.
echo ERROR: Could not fetch or verify app source from Codeberg, GitHub, or Google Drive.
echo Existing app files, models, and outputs have not been deleted.
pause
exit /b 1

:MODEL_FAILED
echo.
echo ERROR: A required model download or size check failed.
echo Existing files were kept. Accept model access and run:
echo   "%HF%" auth login
echo Then rerun installer.bat to resume missing downloads.
pause
exit /b 1

:FAILED
echo.
echo ERROR: Installation stopped. Existing files were kept.
pause
exit /b 1
