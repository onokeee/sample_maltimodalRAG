@echo off
echo ========================================
echo  Multimodal RAG v2.1 
echo  with Image Embedding in Text
echo ========================================
echo.

if not exist venv (
    echo [ERROR] Virtual environment not found
    echo Please run setup.bat first
    echo.
    pause
    exit /b 1
)

echo [PROCESS] Activating virtual environment...
call venv\Scripts\activate.bat

echo [PROCESS] Starting multimodal application...
echo.
streamlit run app_multimodal.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    pause
)
