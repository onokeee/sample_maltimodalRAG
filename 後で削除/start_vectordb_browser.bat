@echo off
echo ========================================
echo  VectorDB Browser v2.2
echo  View Text-Image Relationships
echo ========================================
echo.

if not exist venv (
    echo [ERROR] Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo [PROCESS] Activating virtual environment...
call venv\Scripts\activate.bat

echo [PROCESS] Starting application with VectorDB Browser...
echo.
streamlit run app_vectordb_browser.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    pause
)
