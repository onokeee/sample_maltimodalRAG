# セットアップガイド

このガイドでは、マルチモーダルRAGシステムのセットアップから起動までの手順を説明します。

## 📋 前提条件

- Python 3.8以上がインストールされていること
- OpenAI APIキーを持っていること（https://platform.openai.com/api-keys から取得）

## 🚀 クイックスタート

### 1. 依存パッケージのインストール

プロジェクトディレクトリで以下のコマンドを実行：

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定（オプション）

`.env.example`を`.env`にコピー：

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

`.env`ファイルを開いて、OpenAI APIキーを設定：

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**注意**: `.env`ファイルを作成しなくても、アプリ起動後にサイドバーからAPIキーを入力できます。

### 3. アプリケーションの起動

以下のいずれかの方法で起動：

#### 方法A: 起動スクリプトを使用（推奨）

```bash
python run.py
```

環境チェックが自動的に実行され、問題がなければStreamlitアプリが起動します。

#### 方法B: 直接起動

```bash
streamlit run app.py
```

### 4. ブラウザでアクセス

ブラウザが自動的に開き、`http://localhost:8501`でアプリケーションが表示されます。

## 📖 使い方

### ステップ1: APIキーの設定

サイドバーに表示される「OpenAI APIキー」欄にAPIキーを入力します。
（`.env`ファイルで設定済みの場合は自動的に入力されています）

### ステップ2: ドキュメントのアップロード

1. 「📚 ドキュメント管理」タブを開く
2. 「ファイルをアップロード」からファイルを選択
   - サポート形式: .txt、.pdf、.md
   - 複数ファイル同時アップロード可能
3. 「🔨 インデックスを作成」ボタンをクリック

サンプルドキュメントがすでに`uploaded_data/`フォルダに含まれているので、すぐにインデックス作成を試せます。

### ステップ3: 質問応答

1. 「💬 質問応答」タブを開く
2. チャット入力欄に質問を入力
3. Enterキーを押すと、AIが回答を生成
4. 「参照元ドキュメント」をクリックすると、回答の根拠となったドキュメントを確認できます

## 💡 サンプル質問

インデックス作成後、以下のような質問を試してみてください：

- 「LlamaIndexとは何ですか？」
- 「RAGのメリットを教えてください」
- 「Streamlitでセッション状態を管理する方法は？」
- 「ChromaDBの特徴をまとめてください」

## 🔧 トラブルシューティング

### パッケージのインストールエラー

```bash
# pipをアップグレード
pip install --upgrade pip

# 個別にインストール
pip install streamlit llama-index chromadb openai
```

### APIキーエラー

```
Error: Incorrect API key provided
```

→ OpenAI APIキーが正しいか確認してください
→ APIキーの先頭に余分なスペースがないか確認してください

### ポート8501が使用中エラー

```bash
# 別のポートで起動
streamlit run app.py --server.port 8502
```

### ChromaDBエラー

サイドバーの「🗑️ 全データをリセット」ボタンをクリックして、データベースをリセットしてください。

## 📁 プロジェクト構造

```
multimodal_rag/
├── app.py                      # メインアプリケーション
├── run.py                      # 起動スクリプト
├── requirements.txt            # 依存パッケージ
├── .env.example               # 環境変数サンプル
├── .gitignore                 # Git除外設定
├── README.md                  # プロジェクト説明
├── SETUP.md                   # このファイル
├── uploaded_data/             # アップロードファイル保存先
│   ├── sample_document.txt    # サンプルドキュメント1
│   └── streamlit_guide.md     # サンプルドキュメント2
└── chroma_db/                 # ベクトルDB保存先（自動生成）
```

## 🎯 次のステップ

1. **独自のドキュメントを追加**: `uploaded_data/`フォルダに自分のドキュメントを追加
2. **設定のカスタマイズ**: `app.py`でLLMモデルや検索パラメータを調整
3. **機能の拡張**: 画像サポート、要約機能、エクスポート機能などを追加

## 📞 サポート

問題が発生した場合は、以下を確認してください：

1. Pythonバージョン（3.8以上）
2. 依存パッケージのインストール状態
3. OpenAI APIキーの有効性
4. エラーメッセージの内容

それでも解決しない場合は、GitHubのIssueで報告してください。

## 🔒 セキュリティ注意事項

- `.env`ファイルは絶対にGitにコミットしないでください
- APIキーは他人と共有しないでください
- アップロードしたドキュメントは`uploaded_data/`に保存されます

## 📝 ライセンス

MIT License

---

**Happy Coding! 🎉**
