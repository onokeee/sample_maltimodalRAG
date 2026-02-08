#!/bin/bash

echo "========================================"
echo " マルチモーダルRAGシステム セットアップ"
echo "========================================"
echo

# 仮想環境の存在チェック
if [ -d "venv" ]; then
    echo "[情報] 既存の仮想環境が見つかりました"
    read -p "既存の仮想環境を削除して再作成しますか？ (y/N): " answer
    case $answer in
        [Yy]* )
            echo
            echo "[処理] 既存の仮想環境を削除中..."
            rm -rf venv
            echo "[完了] 既存の仮想環境を削除しました"
            ;;
        * )
            echo "[処理] 既存の仮想環境を使用します"
            ;;
    esac
fi

# 仮想環境の作成
if [ ! -d "venv" ]; then
    echo
    echo "[処理] Python仮想環境を作成中..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[エラー] 仮想環境の作成に失敗しました"
        echo "Python3が正しくインストールされているか確認してください"
        exit 1
    fi
    echo "[完了] 仮想環境を作成しました"
fi

# 仮想環境のアクティベート
echo
echo "[処理] 仮想環境をアクティベート中..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[エラー] 仮想環境のアクティベートに失敗しました"
    exit 1
fi
echo "[完了] 仮想環境をアクティベートしました"

# pipのアップグレード
echo
echo "[処理] pipをアップグレード中..."
pip install --upgrade pip

# パッケージのインストール
echo
echo "[処理] 必要なパッケージをインストール中..."
echo "この処理には数分かかる場合があります..."
echo
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo
    echo "[エラー] パッケージのインストールに失敗しました"
    exit 1
fi

echo
echo "========================================"
echo " セットアップ完了！"
echo "========================================"
echo
echo "次のステップ:"
echo "1. .env.example を .env にコピーして、APIキーを設定"
echo "   cp .env.example .env"
echo "2. ./start.sh を実行してアプリを起動"
echo
echo "または、以下のコマンドで起動:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo
