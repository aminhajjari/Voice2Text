@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"

echo Checking Python installation...
set PY_CMD=
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PY_CMD=py -3
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set PY_CMD=python
    )
)

if "%PY_CMD%"=="" (
    echo Python not found on PATH.
    echo Please install Python from python.org and ensure it is added to your PATH.
    pause
    exit /b 1
)

set VENV_PYTHON=.venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment...
    %PY_CMD% -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

if exist requirements.txt (
    set REQ_HASH=
    for /f %%H in ('%VENV_PYTHON% -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path(r'requirements.txt').read_bytes()).hexdigest())"') do set REQ_HASH=%%H
    set STORED_REQ_HASH=
    if exist .venv\requirements.sha256 (
        set /p STORED_REQ_HASH=<.venv\requirements.sha256
    )
    if not defined REQ_HASH (
        echo Failed to calculate requirements hash.
        pause
        exit /b 1
    )
    if /I not "!REQ_HASH!"=="!STORED_REQ_HASH!" (
        echo Installing dependencies...
        "%VENV_PYTHON%" -m pip install --upgrade pip
        "%VENV_PYTHON%" -m pip install -r requirements.txt
        if %ERRORLEVEL% equ 0 (
            > .venv\requirements.sha256 echo !REQ_HASH!
        )
    )
)

echo Running transcription app...

if exist transcribe.py (
    "%VENV_PYTHON%" transcribe.py %*
) else if exist main.py (
    "%VENV_PYTHON%" main.py %*
) else (
    echo Could not find the Python entry point.
    echo Expected transcribe.py or main.py.
    pause
    exit /b 1
)

set EXITCODE=%ERRORLEVEL%

if "%EXITCODE%"=="0" (
    echo Transcription completed successfully.
) else (
    echo Transcription finished with errors ^(exit code %EXITCODE%^).
    pause
)

popd
exit /b %EXITCODE%
