@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE="
set "BASE_PYTHON="
set "APP_FILE=%~dp0app.py"
set "REQ_FILE=%~dp0requirements.txt"
set "LOCAL_HOME=%~dp0.runtime_home"
set "LOCAL_STREAMLIT=%LOCAL_HOME%\.streamlit"
set "LOG_DIR=%~dp0logs"
set "LOG_FILE=%LOG_DIR%\streamlit.log"

echo ==========================================
echo Crop Yield Planner - Setup and Launch
echo ==========================================
echo.

call :find_python
if not defined BASE_PYTHON (
    echo No usable Python installation was found.
    echo.
    echo Please install Python 3.10 or newer from:
    echo https://www.python.org/downloads/windows/
    echo.
    echo During install, enable:
    echo - Add python.exe to PATH
    echo.
    echo If Windows is redirecting to the Microsoft Store:
    echo Settings ^> Apps ^> Advanced app settings ^> App execution aliases
    echo Then turn off the aliases for python.exe and python3.exe
    echo.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    call :run_base_python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%LOCAL_STREAMLIT%" (
    mkdir "%LOCAL_STREAMLIT%"
)

if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%"
)

set "HOME=%LOCAL_HOME%"
set "USERPROFILE=%LOCAL_HOME%"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

if exist "%REQ_FILE%" (
    echo Installing project requirements...
    "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
    if errorlevel 1 (
        echo Failed to install the required packages.
        pause
        exit /b 1
    )
) else (
    echo requirements.txt was not found.
    pause
    exit /b 1
)

echo Starting Streamlit app...
if exist "%LOG_FILE%" del "%LOG_FILE%"
start "Crop Yield Planner" cmd /c ""%PYTHON_EXE%" -m streamlit run "%APP_FILE%" --server.headless true --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false > "%LOG_FILE%" 2>&1"

echo Waiting for the local server...
call :wait_for_streamlit
if errorlevel 1 (
    echo.
    echo Streamlit did not become available on http://localhost:8501
    echo Check the log file for the startup error:
    echo %LOG_FILE%
    echo.
    if exist "%LOG_FILE%" type "%LOG_FILE%"
    echo.
    pause
    exit /b 1
)

echo Opening the app in your browser...
start "" http://localhost:8501

echo.
echo If the browser does not open automatically, visit:
echo http://localhost:8501
echo.
echo Note: If the trained model .pkl files are missing, the app
echo will still run using historical district-based recommendations.
echo.
pause
exit /b 0

:find_python
call :test_python_command py -3
if defined BASE_PYTHON exit /b 0
call :test_python_command py
if defined BASE_PYTHON exit /b 0
call :test_python_command python
if defined BASE_PYTHON exit /b 0

for %%P in (
    "%LocalAppData%\Programs\Python\Python313\python.exe"
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "%LocalAppData%\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python313\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python310\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    call :test_python_path "%%~P"
    if defined BASE_PYTHON exit /b 0
)
exit /b 0

:test_python_command
set "CANDIDATE=%*"
%CANDIDATE% --version >nul 2>nul
if errorlevel 1 exit /b 0

set "BASE_PYTHON=%CANDIDATE%"
exit /b 0

:test_python_path
set "CANDIDATE=%~1"
if not exist "%CANDIDATE%" exit /b 0

"%CANDIDATE%" --version >nul 2>nul
if errorlevel 1 exit /b 0

set "BASE_PYTHON=%CANDIDATE%"
exit /b 0

:run_base_python
%BASE_PYTHON% %*
exit /b %errorlevel%

:wait_for_streamlit
set /a ATTEMPTS=0
:wait_loop
set /a ATTEMPTS+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest 'http://127.0.0.1:8501' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 exit /b 0
if %ATTEMPTS% GEQ 15 exit /b 1
timeout /t 2 /nobreak >nul
goto wait_loop
