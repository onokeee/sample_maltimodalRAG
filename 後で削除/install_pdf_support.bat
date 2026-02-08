@echo off
echo ========================================
echo  PDF Image Support Installation
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

echo [PROCESS] Installing PDF and image support packages...
echo.
pip install pypdf pdfminer.six pdfplumber pymupdf llama-index-readers-file pdf2image

echo.
echo [INFO] pdf2image requires Poppler to be installed.
echo.
echo Please download Poppler for Windows from:
echo https://github.com/oschwartz10612/poppler-windows/releases/
echo.
echo Extract and add the 'bin' folder to your PATH, or
echo place poppler files in your project directory.
echo.

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install packages
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo PDF files with images should now work correctly.
echo Please restart the application.
echo.
pause
