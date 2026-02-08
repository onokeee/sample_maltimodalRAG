# Python バージョンについて

## ✅ 対応バージョン

このプロジェクトは **Python 3.8以上** に対応しています。

- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12
- ✅ Python 3.13

## 🔍 現在のPythonバージョンを確認

```bash
python --version
```

または

```bash
python3 --version
```

## 📦 バージョン別の注意事項

### Python 3.12以上を使用している場合

**Good News!** このプロジェクトは最新のPython 3.12+に完全対応しています。

requirements.txtは自動的に互換性のあるパッケージバージョンをインストールします：

```
llama-index>=0.11.0           # Python 3.12+対応
llama-index-vector-stores-chroma
llama-index-embeddings-openai
llama-index-llms-openai
streamlit>=1.31.0
chromadb>=0.4.22
openai>=1.12.0
pillow>=10.0.0
pypdf>=4.0.0
python-dotenv>=1.0.0
```

### Python 3.8〜3.11を使用している場合

問題なく動作します。最新版のパッケージが自動的にインストールされます。

## 🐛 トラブルシューティング

### エラー: "Requires-Python >=3.8.1,<3.12"

このエラーが表示される場合、古いバージョンのrequirements.txtが使用されている可能性があります。

**解決方法:**

1. requirements.txtが最新版であることを確認
2. venv環境を再作成

```bash
# Windows
rmdir /s /q venv
setup.bat

# Mac/Linux
rm -rf venv
./setup.sh
```

### エラー: "Could not find a version that satisfies the requirement"

**解決方法:**

```bash
# venv環境をアクティベート
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# pipをアップグレード
python -m pip install --upgrade pip

# パッケージを再インストール
pip install -r requirements.txt
```

### 特定のパッケージでエラーが発生する場合

**個別インストールを試す:**

```bash
pip install llama-index
pip install streamlit
pip install chromadb
pip install openai
pip install pillow pypdf python-dotenv
```

## 🔄 Pythonバージョンの変更

### 異なるPythonバージョンで実行したい場合

```bash
# 特定のPythonバージョンでvenvを作成
python3.11 -m venv venv  # Python 3.11を使用
python3.12 -m venv venv  # Python 3.12を使用

# アクティベート
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# パッケージインストール
pip install -r requirements.txt
```

## 📊 推奨環境

| 項目 | 推奨 |
|------|------|
| Python | 3.10以上 |
| pip | 最新版 |
| OS | Windows 10/11, macOS 10.15+, Ubuntu 20.04+ |

## 💡 ヒント

### pyenvを使用した複数バージョン管理

複数のPythonバージョンを管理したい場合は、pyenvの使用をおすすめします：

```bash
# pyenvでPython 3.12をインストール
pyenv install 3.12.0

# プロジェクトでPython 3.12を使用
pyenv local 3.12.0

# venv作成
python -m venv venv
```

### Anacondaを使用している場合

```bash
# 新しい環境を作成
conda create -n multimodal_rag python=3.12

# 環境をアクティベート
conda activate multimodal_rag

# パッケージインストール
pip install -r requirements.txt
```

## 📞 サポート

Pythonバージョンに関する問題が解決しない場合は、以下の情報を含めてGitHubのIssueで報告してください：

1. Pythonバージョン (`python --version`)
2. OSとバージョン
3. エラーメッセージの全文
4. 実行したコマンド

---

**最新版のrequirements.txtを使用していれば、Python 3.8〜3.13まで問題なく動作します！**
