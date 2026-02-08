#!/bin/bash

echo "========================================"
echo "Multimodal RAG System v2.0 - Improved"
echo "========================================"
echo ""

# 仮想環境の確認
if [ ! -f "venv/bin/activate" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run ./setup.sh first."
    exit 1
fi

# 仮想環境の有効化
source venv/bin/activate

# 改善版アプリの起動
echo "Starting improved application..."
echo ""
streamlit run app_improved.py
