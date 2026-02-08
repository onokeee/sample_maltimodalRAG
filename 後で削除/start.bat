@echo off
echo ========================================
echo  Multimodal RAG System Start
echo ========================================
echo.

REM Check if venv exists
if not exist venv (
    echo [ERROR] Virtual environment not found
    echo Please run setup.bat first
    echo.
    pause
    exit /b 1
)

echo [PROCESS] Activating virtual environment...
call venv\Scripts\activate.bat

echo [PROCESS] Checking environment...
python run.py

REM If error occurred
if errorlevel 1 (
    echo.
    echo [INFO] An error occurred
    echo Troubleshooting:
    echo 1. Try running setup.bat again
    echo 2. Check if API key is set in .env file
    echo 3. Or enter API key in sidebar after app starts
    echo.
    pause
)
