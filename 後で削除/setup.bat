@echo off
echo ========================================
echo  Multimodal RAG System Setup
echo ========================================
echo.

REM Check if venv exists
if exist venv (
    echo [INFO] Existing virtual environment found
    choice /C YN /M "Delete and recreate the virtual environment?"
    if errorlevel 2 goto activate_existing
    if errorlevel 1 goto delete_venv
) else (
    goto create_venv
)

:delete_venv
echo.
echo [PROCESS] Deleting existing virtual environment...
rmdir /s /q venv
echo [DONE] Deleted existing virtual environment
goto create_venv

:create_venv
echo.
echo [PROCESS] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    echo Please check if Python is installed correctly
    pause
    exit /b 1
)
echo [DONE] Virtual environment created
goto activate_and_install

:activate_existing
echo [PROCESS] Using existing virtual environment
goto activate_and_install

:activate_and_install
echo.
echo [PROCESS] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [DONE] Virtual environment activated
echo.
echo [PROCESS] Upgrading pip...
python -m pip install --upgrade pip
echo.
echo [PROCESS] Installing required packages...
echo This may take a few minutes...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install packages
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env and set your API key
echo 2. Run start.bat to launch the application
echo.
echo Or run manually:
echo   venv\Scripts\activate
echo   streamlit run app.py
echo.
pause
