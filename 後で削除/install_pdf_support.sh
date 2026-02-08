#!/bin/bash

echo "========================================"
echo " Additional Packages Installation"
echo "========================================"
echo

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found"
    echo "Please run ./setup.sh first"
    echo
    exit 1
fi

# Activate virtual environment
echo "[PROCESS] Activating virtual environment..."
source venv/bin/activate

# Install PDF support packages
echo "[PROCESS] Installing PDF support packages..."
echo
pip install pypdf pdfminer.six pdfplumber pymupdf llama-index-readers-file

if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Failed to install packages"
    exit 1
fi

echo
echo "========================================"
echo " Installation Complete!"
echo "========================================"
echo
echo "PDF files should now work correctly."
echo "Please restart the application."
echo
