# 🚀 クイックスタートガイド

## 最速で起動する3ステップ

### 1️⃣ パッケージをインストール

```bash
cd C:\MCP_Learning\chapter01\multimodal_rag
pip install -r requirements.txt
```

### 2️⃣ アプリを起動

```bash
python run.py
```

または

```bash
streamlit run app.py
```

### 3️⃣ ブラウザでアクセス

自動的に `http://localhost:8501` が開きます

---

## 初回セットアップ

### APIキーの設定

サイドバーに**OpenAI APIキー**を入力
- まだ持っていない場合: https://platform.openai.com/api-keys

### すぐに試す

1. 「📚 ドキュメント管理」タブ
2. 「🔨 インデックスを作成」をクリック
   - サンプルドキュメントが既に用意されています！
3. 「💬 質問応答」タブ
4. 質問してみる：
   - 「LlamaIndexとは何ですか？」
   - 「RAGのメリットを教えてください」

---

## トラブル時

### パッケージエラー
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### ポート使用中
```bash
streamlit run app.py --server.port 8502
```

### データリセット
アプリのサイドバー → 「🗑️ 全データをリセット」

---

**詳細は `SETUP.md` を参照してください！**
