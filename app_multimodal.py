"""
メインアプリケーション - マルチモーダル専用版 v3.0
🆕 通常モード削除、常にGPT-4 Visionで最高品質の回答
"""
import streamlit as st
import chromadb
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# カスタムモジュール  
from core.rag_engine import initialize_rag_system, load_and_index_documents
from core.image_handler import ImageCache
from core.multimodal_query import query_with_multimodal, render_response_with_images
from core.vectordb_browser import render_vectordb_browser, export_vectordb_summary, get_all_documents_from_vectordb
from utils.logger import get_logger
from utils.exceptions import (
    APIKeyError, FileUploadError, IndexCreationError, 
    QueryError, PDFProcessingError
)

# LlamaIndexインポート（インデックス自動ロード用）
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# 環境変数の読み込み
load_dotenv()

# ロガー初期化
logger = get_logger()
logger.info("=" * 50)
logger.info("Application started - Multimodal Only v3.0")
logger.info("=" * 50)

# ページ設定
st.set_page_config(
    page_title="マルチモーダルRAGシステム v3.0",
    page_icon="🔍",
    layout="wide"
)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_created" not in st.session_state:
    st.session_state.index_created = False
if "image_cache" not in st.session_state:
    st.session_state.image_cache = ImageCache()
if "embed_model" not in st.session_state:
    st.session_state.embed_model = "text-embedding-3-small"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-4o-mini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1
if "use_chat_history" not in st.session_state:
    st.session_state.use_chat_history = True
if "chat_history_length" not in st.session_state:
    st.session_state.chat_history_length = 5
if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0
if "frequency_penalty" not in st.session_state:
    st.session_state.frequency_penalty = 0.0
if "presence_penalty" not in st.session_state:
    st.session_state.presence_penalty = 0.0
if "seed" not in st.session_state:
    st.session_state.seed = None

# 🆕 起動時にインデックスと画像キャッシュを自動ロード
if "index" not in st.session_state:
    try:
        chroma_dir = Path("./chroma_db")
        if chroma_dir.exists() and list(chroma_dir.glob("*")):
            # ChromaDBにデータが存在する場合、インデックスをロード
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            collections = chroma_client.list_collections()
            
            if collections:
                # 🔧 LlamaIndex Settings を初期化（重要！）
                Settings.embed_model = OpenAIEmbedding(model=st.session_state.embed_model)
                Settings.llm = LlamaOpenAI(model=st.session_state.llm_model, temperature=st.session_state.temperature)
                logger.info(f"LlamaIndex Settings initialized: embed={st.session_state.embed_model}, llm={st.session_state.llm_model}")
                
                # インデックスを再構築
                collection = collections[0]
                vector_store = ChromaVectorStore(chroma_collection=collection)
                st.session_state.index = VectorStoreIndex.from_vector_store(vector_store)
                st.session_state.index_created = True
                logger.info(f"✅ Index auto-loaded from ChromaDB: {collection.name} ({collection.count()} documents)")
                
                # 🆕 画像キャッシュを復元（VectorDBのメタデータから）
                try:
                    results = collection.get(include=["metadatas"])
                    image_cache = st.session_state.image_cache
                    restored_count = 0
                    
                    for metadata in results.get("metadatas", []):
                        if "image_ids" in metadata:
                            import json
                            try:
                                if isinstance(metadata["image_ids"], str):
                                    image_ids = json.loads(metadata["image_ids"])
                                else:
                                    image_ids = metadata["image_ids"]
                                
                                # 各image_idをレジストリに登録
                                for image_id in image_ids:
                                    cache_path = image_cache._get_cache_path(image_id)
                                    if cache_path.exists() and image_id not in image_cache.registry:
                                        file_size = cache_path.stat().st_size
                                        image_cache.registry[image_id] = {
                                            "path": str(cache_path),
                                            "metadata": {
                                                "file_name": metadata.get("file_name"),
                                                "page": metadata.get("page"),
                                                "type": "cached"
                                            },
                                            "size": file_size
                                        }
                                        image_cache.current_memory += file_size
                                        restored_count += 1
                            except (json.JSONDecodeError, TypeError) as e:
                                continue
                    
                    logger.info(f"✅ Image cache restored: {restored_count} images")
                except Exception as e:
                    logger.warning(f"Failed to restore image cache: {e}")
            else:
                st.session_state.index = None
                logger.info("No collections found in ChromaDB")
        else:
            st.session_state.index = None
            logger.info("ChromaDB directory empty or not found")
    except Exception as e:
        logger.error(f"Failed to auto-load index: {e}", exc_info=True)
        st.session_state.index = None


@st.cache_resource
def get_chroma_client():
    """ChromaDBクライアントの取得"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        logger.info("ChromaDB client initialized")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        st.error(f"データベースの初期化に失敗しました: {e}")
        return None


def validate_api_key(api_key):
    """APIキーの検証"""
    if not api_key:
        raise APIKeyError("APIキーが入力されていません")
    
    if not api_key.startswith("sk-"):
        raise APIKeyError("無効なAPIキー形式です（sk-で始まる必要があります）")
    
    if len(api_key) < 20:
        raise APIKeyError("APIキーが短すぎます")
    
    logger.info("API key validated successfully")
    return True


def validate_file_upload(uploaded_file):
    """アップロードファイルの検証"""
    max_size_mb = 100
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        raise FileUploadError(
            f"ファイルサイズが大きすぎます: {file_size_mb:.1f}MB（上限: {max_size_mb}MB）"
        )
    
    allowed_types = ['.txt', '.pdf', '.md']
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    if file_ext not in allowed_types:
        raise FileUploadError(
            f"サポートされていないファイル形式です: {file_ext}（対応形式: {', '.join(allowed_types)}）"
        )
    
    logger.info(f"File upload validated: {uploaded_file.name} ({file_size_mb:.1f}MB)")
    return True


# メインUI
st.title("🔍 マルチモーダルRAGシステム v3.0")
st.caption("🤖 GPT-4 Vision | 🖼️ 画像理解 | 🔍 VectorDBブラウザー | ⚡ 並列処理")
st.markdown("---")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    api_key_input = st.text_input(
        "OpenAI APIキー",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI APIキーを入力してください"
    )
    
    if api_key_input:
        try:
            validate_api_key(api_key_input)
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.success("✅ APIキーが設定されました")
            logger.info("API key configured")
        except APIKeyError as e:
            st.error(f"❌ {str(e)}")
            logger.warning(f"Invalid API key: {e}")
    else:
        st.warning("⚠️ APIキーを入力してください")
    
    st.markdown("---")
    
    st.subheader("🤖 モデル設定")
    
    # Embeddingモデル選択
    embed_model = st.selectbox(
        "Embeddingモデル",
        options=[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ],
        index=0,
        help="テキストのベクトル化に使用するモデル"
    )
    
    # LLMモデル選択
    llm_model = st.selectbox(
        "LLMモデル（回答生成）",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        index=0,
        help="質問応答に使用するモデル（画像理解も含む）"
    )
    
    # Temperature設定
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1,
        help="0=一貫性重視、2=創造性重視"
    )
    
    # Top P (nucleus sampling)
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="累積確率のしきい値。低いほど保守的な出力"
    )
    
    # Frequency Penalty
    frequency_penalty = st.slider(
        "Frequency Penalty",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="繰り返しを減らす。高いほど新しい表現を使用"
    )
    
    # Presence Penalty
    presence_penalty = st.slider(
        "Presence Penalty",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="新しいトピックを促進。高いほど多様性が増す"
    )
    
    # Seed（再現性）
    use_seed = st.checkbox("Seed固定（再現性）", value=False)
    seed = None
    if use_seed:
        seed = st.number_input(
            "Seed値",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            help="同じSeedで同じ結果を再現"
        )
    
    # セッション状態に保存
    st.session_state.embed_model = embed_model
    st.session_state.llm_model = llm_model
    st.session_state.temperature = temperature
    st.session_state.top_p = top_p
    st.session_state.frequency_penalty = frequency_penalty
    st.session_state.presence_penalty = presence_penalty
    st.session_state.seed = seed
    
    st.markdown("---")
    
    st.info("🤖 **GPT-4 Vision**: 画像を理解して文章中に埋め込みます")
    
    st.markdown("---")
    
    st.subheader("🎨 画像抽出設定")
    
    extraction_method = st.selectbox(
        "抽出方法",
        options=["high_quality", "medium_quality", "embedded", "combined"],
        format_func=lambda x: {
            "high_quality": "高品質（ページ全体）",
            "medium_quality": "中品質（ページ全体）",
            "embedded": "埋め込み画像（位置ベース）",
            "combined": "全て（ページ+埋め込み）"
        }[x],
        index=0
    )
    
    if extraction_method in ["high_quality", "medium_quality", "combined"]:
        dpi = st.slider("解像度（DPI）", 72, 300, 200, 50)
    else:
        dpi = 150
    
    max_workers = st.slider(
        "並列処理スレッド数",
        min_value=1,
        max_value=5,
        value=3,
        help="PDFファイルの処理を並列化します（速度向上）"
    )
    
    st.markdown("---")
    
    st.subheader("💬 質問応答設定")
    
    # 会話履歴を使用
    use_chat_history = st.checkbox(
        "会話履歴を考慣",
        value=True,
        help="過去の会話を考慣した回答を生成"
    )
    
    # 会話履歴の長さ
    chat_history_length = 5
    if use_chat_history:
        chat_history_length = st.slider(
            "会話履歴数",
            min_value=1,
            max_value=20,
            value=5,
            help="何ターン分の会話を考慣するか"
        )
    
    # 検索結果件数
    similarity_top_k = st.slider(
        "検索結果件数",
        min_value=1,
        max_value=10,
        value=3,
        help="類似度が高い上位N件を取得"
    )
    
    # Max Tokens
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=2000,
        step=100,
        help="回答の最大トークン数"
    )
    
    # 画像詳細レベル
    image_detail = st.selectbox(
        "画像詳細レベル",
        options=["high", "low", "auto"],
        index=0,
        help="high=高画質、low=低コスト、auto=自動"
    )
    
    # 最大画像数
    max_images = st.slider(
        "最大画像数",
        min_value=1,
        max_value=10,
        value=5,
        help="1回の質問で送信する画像の最大数"
    )
    
    # Response Mode
    response_mode = st.selectbox(
        "Response Mode",
        options=["compact", "tree_summarize", "simple_summarize"],
        index=0,
        help="compact=コンパクト、tree_summarize=階層要約、simple_summarize=シンプル要約"
    )
    
    # セッション状態に保存
    st.session_state.use_chat_history = use_chat_history
    st.session_state.chat_history_length = chat_history_length
    st.session_state.max_tokens = max_tokens
    st.session_state.image_detail = image_detail
    st.session_state.max_images = max_images
    st.session_state.response_mode = response_mode
    
    st.markdown("---")
    
    st.subheader("📊 統計情報")
    data_dir = Path("./uploaded_data")
    if data_dir.exists():
        files = list(data_dir.glob("*.*"))
        st.metric("ファイル", len(files))
    else:
        st.metric("ファイル", 0)
    
    total_images = len(st.session_state.image_cache.registry)
    if total_images > 0:
        st.metric("画像キャッシュ", total_images)
        cache_size_mb = st.session_state.image_cache.current_memory / (1024 * 1024)
        st.caption(f"使用量: {cache_size_mb:.1f}MB")
    
    if st.session_state.index_created:
        st.success("✅ インデックス作成済み")
    else:
        st.info("ℹ️ インデックス未作成")
    
    st.markdown("---")
    
    st.subheader("👁️ 表示設定")
    show_sources = st.checkbox("参照元を表示", value=True)
    
    st.markdown("---")
    
    if st.button("🗑️ 全データをリセット", type="secondary"):
        if st.session_state.get("confirm_reset", False):
            try:
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                    data_dir.mkdir()
                
                chroma_dir = Path("./chroma_db")
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir)
                    chroma_dir.mkdir()
                
                st.session_state.image_cache.clear()
                st.session_state.index_created = False
                st.session_state.messages = []
                st.cache_resource.clear()
                
                st.success("✅ リセット完了")
                logger.info("All data reset successfully")
                st.session_state.confirm_reset = False
                st.rerun()
            except Exception as e:
                st.error(f"❌ リセット失敗: {e}")
                logger.error(f"Reset failed: {e}")
        else:
            st.session_state.confirm_reset = True
            st.warning("⚠️ もう一度クリックして確認")

# メインコンテンツ
if not api_key_input:
    st.info("👈 サイドバーからOpenAI APIキーを入力してください")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 ドキュメント管理", 
    "💬 質問応答", 
    "🔍 VectorDBブラウザー",
    "📊 システム情報"
])

# タブ1: ドキュメント管理
with tab1:
    st.header("📚 ドキュメント管理")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "ファイルをアップロード（複数選択可）",
            accept_multiple_files=True,
            type=["txt", "pdf", "md"],
            help="対応形式: .txt, .pdf, .md（最大100MB/ファイル）"
        )
        
        if uploaded_files:
            data_dir = Path("./uploaded_data")
            data_dir.mkdir(exist_ok=True)
            
            success_count = 0
            error_count = 0
            
            for uploaded_file in uploaded_files:
                try:
                    validate_file_upload(uploaded_file)
                    file_path = data_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    success_count += 1
                    logger.info(f"File uploaded: {uploaded_file.name}")
                except FileUploadError as e:
                    st.error(f"❌ {uploaded_file.name}: {str(e)}")
                    error_count += 1
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name}: 予期しないエラー - {str(e)}")
                    error_count += 1
            
            if success_count > 0:
                st.success(f"✅ {success_count}件保存完了")
            if error_count > 0:
                st.warning(f"⚠️ {error_count}件失敗")
    
    with col2:
        st.subheader("📁 ファイル管理")
        data_dir = Path("./uploaded_data")
        if data_dir.exists():
            files = sorted(data_dir.glob("*.*"))
            if files:
                st.write(f"**ファイル数: {len(files)}**")
                
                # ファイル削除モード
                if st.checkbox("🗑️ ファイル削除モード", key="file_delete_mode"):
                    st.warning("削除したいファイルを選択してください")
                    
                    files_to_delete = []
                    for file in files:
                        size_kb = file.stat().st_size / 1024
                        icon = "📄" if file.suffix.lower() == ".pdf" else "📝"
                        
                        if st.checkbox(
                            f"{icon} {file.name} ({size_kb:.1f}KB)",
                            key=f"del_{file.name}"
                        ):
                            files_to_delete.append(file)
                    
                    if files_to_delete:
                        col_del1, col_del2 = st.columns(2)
                        with col_del1:
                            if st.button("🗑️ 削除実行", type="primary", use_container_width=True):
                                deleted_count = 0
                                for file in files_to_delete:
                                    try:
                                        file.unlink()
                                        deleted_count += 1
                                        logger.info(f"File deleted: {file.name}")
                                    except Exception as e:
                                        st.error(f"削除失敗: {file.name} - {e}")
                                        logger.error(f"Failed to delete file: {file.name} - {e}")
                                
                                if deleted_count > 0:
                                    st.success(f"✅ {deleted_count}ファイルを削除しました")
                                    st.rerun()
                        with col_del2:
                            if st.button("❌ キャンセル", use_container_width=True):
                                st.rerun()
                else:
                    # 通常表示モード
                    for file in files:
                        size_kb = file.stat().st_size / 1024
                        icon = "📄" if file.suffix.lower() == ".pdf" else "📝"
                        st.text(f"{icon} {file.name} ({size_kb:.1f}KB)")
            else:
                st.info("ファイルなし")
        else:
            st.info("ファイルなし")
    
    st.markdown("---")
    
    # インデックス作成モード選択
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        if st.button("🔨 新規作成（上書き）", type="primary", use_container_width=True):
            st.session_state.index_mode = "overwrite"
            st.session_state.selected_files = None
    with col_mode2:
        if st.button("➕ 追加作成", type="secondary", use_container_width=True):
            st.session_state.index_mode = "append"
            st.session_state.selected_files = None
    
    # 追加モードの場合、ファイル選択 UI を表示
    if st.session_state.get("index_mode") == "append" and st.session_state.get("selected_files") is None:
        st.info("📝 追加するファイルを選択してください")
        
        data_dir = Path("./uploaded_data")
        if data_dir.exists():
            all_files = sorted([f for f in data_dir.glob("*.*")])
            
            if all_files:
                st.write("📦 **ファイル一覧**")
                
                selected_files = []
                select_all = st.checkbox("☑️ 全て選択", value=False)
                
                for file in all_files:
                    size_kb = file.stat().st_size / 1024
                    icon = "📄" if file.suffix.lower() == ".pdf" else "📝"
                    default_checked = select_all
                    
                    if st.checkbox(
                        f"{icon} {file.name} ({size_kb:.1f}KB)",
                        value=default_checked,
                        key=f"file_select_{file.name}"
                    ):
                        selected_files.append(file.name)
                
                st.markdown("---")
                
                col_confirm1, col_confirm2 = st.columns(2)
                with col_confirm1:
                    if st.button("✅ 選択したファイルで追加", type="primary", use_container_width=True):
                        if selected_files:
                            st.session_state.selected_files = selected_files
                            st.rerun()
                        else:
                            st.warning("⚠️ ファイルが選択されていません")
                with col_confirm2:
                    if st.button("❌ キャンセル", use_container_width=True):
                        st.session_state.index_mode = None
                        st.rerun()
            else:
                st.warning("⚠️ ファイルがありません")
                st.session_state.index_mode = None
        else:
            st.error("❌ uploaded_dataディレクトリが存在しません")
            st.session_state.index_mode = None
    
    # モードが選択されて、且つ追加モード以外またはファイルが選択済みの場合のみ処理実行
    if "index_mode" in st.session_state and st.session_state.index_mode:
        mode = st.session_state.index_mode
        
        # 追加モードでファイルが未選択の場合はスキップ
        if mode == "append" and st.session_state.get("selected_files") is None:
            pass
        else:
            data_dir = Path("./uploaded_data")
            
            if not data_dir.exists() or not list(data_dir.glob("*.*")):
                st.error("❌ アップロードされたファイルがありません")
                st.session_state.index_mode = None
                st.session_state.selected_files = None
            else:
                try:
                    # 追加モードで選択されたファイルのみを処理
                    if mode == "append" and st.session_state.get("selected_files"):
                        temp_dir = Path("./temp_selected_files")
                        if temp_dir.exists():
                            shutil.rmtree(temp_dir)
                        temp_dir.mkdir()
                        
                        selected_file_names = st.session_state.selected_files
                        for file_name in selected_file_names:
                            src = data_dir / file_name
                            dst = temp_dir / file_name
                            if src.exists():
                                shutil.copy2(src, dst)
                                logger.info(f"Selected for indexing: {file_name}")
                        
                        target_dir = temp_dir
                        logger.info(f"Processing {len(selected_file_names)} selected files")
                    else:
                        target_dir = data_dir
                    
                    mode_label = "新規作成（既存データを削除）" if mode == "overwrite" else f"追加作成（{len(st.session_state.get('selected_files', []))}ファイル）"
                    with st.spinner(f"インデックスを{mode_label}中..."):
                        chroma_client = get_chroma_client()
                        
                        if chroma_client is None:
                            raise IndexCreationError("データベースクライアントの初期化に失敗しました")
                        
                        if mode == "overwrite":
                            storage_context = initialize_rag_system(chroma_client)
                        else:
                            try:
                                collection = chroma_client.get_collection("multimodal_rag")
                                vector_store = ChromaVectorStore(chroma_collection=collection)
                                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                                logger.info("Using existing collection for append mode")
                            except Exception:
                                storage_context = initialize_rag_system(chroma_client)
                                logger.info("No existing collection, creating new one")
                        
                        index, error = load_and_index_documents(
                            str(target_dir),
                            storage_context,
                            extraction_method,
                            dpi,
                            max_workers=max_workers
                        )
                        
                        if mode == "append" and 'temp_dir' in locals() and temp_dir.exists():
                            shutil.rmtree(temp_dir)
                            logger.info("Cleaned up temporary directory")
                        
                        if error:
                            raise IndexCreationError(error)
                        
                        st.session_state.index = index
                        st.session_state.index_created = True
                        
                        success_msg = "✅ インデックスを新規作成しました！" if mode == "overwrite" else f"✅ {len(st.session_state.get('selected_files', []))}ファイルを追加しました！"
                        st.success(success_msg)
                        st.balloons()
                        
                        st.session_state.index_mode = None
                        st.session_state.selected_files = None
                
                except (IndexCreationError, PDFProcessingError) as e:
                    st.error(f"❌ エラー: {str(e)}")
                    st.session_state.index_mode = None
                    st.session_state.selected_files = None
                    if 'temp_dir' in locals() and temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    st.error(f"❌ 予期しないエラー: {str(e)}")
                    with st.expander("詳細エラー情報"):
                        st.code(str(e))
                    st.session_state.index_mode = None
                    st.session_state.selected_files = None
                    if 'temp_dir' in locals() and temp_dir.exists():
                        shutil.rmtree(temp_dir)
    
    st.markdown("---")
    
    # インデックス削除機能
    st.markdown("---")
    
    # インデックス管理機能
    if st.session_state.index_created:
        st.subheader("🗑️ インデックス管理")
        
        # タブで機能を分ける
        idx_tab1, idx_tab2 = st.tabs(["個別削除", "全削除"])
        
        with idx_tab1:
            st.write("**インデックスから削除するファイルを選択**")
            
            # VectorDBからファイル一覧を取得
            chroma_client = get_chroma_client()
            if chroma_client:
                try:
                    collection = chroma_client.get_collection("multimodal_rag")
                    results = collection.get(include=["metadatas"])
                    
                    # ファイル名でグループ化
                    file_dict = {}
                    for idx, metadata in enumerate(results.get("metadatas", [])):
                        file_name = metadata.get("file_name", "Unknown")
                        if file_name not in file_dict:
                            file_dict[file_name] = []
                        file_dict[file_name].append(results["ids"][idx])
                    
                    if file_dict:
                        files_to_delete = []
                        for file_name, doc_ids in sorted(file_dict.items()):
                            chunk_count = len(doc_ids)
                            if st.checkbox(
                                f"📄 {file_name} ({chunk_count}チャンク)",
                                key=f"idx_del_{file_name}"
                            ):
                                files_to_delete.append((file_name, doc_ids))
                        
                        if files_to_delete:
                            st.markdown("---")
                            col_idx1, col_idx2 = st.columns(2)
                            with col_idx1:
                                if st.button("🗑️ 選択ファイルのインデックスを削除", type="primary", use_container_width=True):
                                    try:
                                        deleted_files = []
                                        for file_name, doc_ids in files_to_delete:
                                            # ChromaDBから削除
                                            collection.delete(ids=doc_ids)
                                            deleted_files.append(file_name)
                                            logger.info(f"Index deleted for file: {file_name} ({len(doc_ids)} chunks)")
                                        
                                        # 画像キャッシュから該当ファイルの画像を削除
                                        image_cache = st.session_state.image_cache
                                        for file_name in deleted_files:
                                            # レジストリから該当ファイルの画像を探して削除
                                            images_to_remove = []
                                            for image_id, info in image_cache.registry.items():
                                                if info["metadata"].get("file_name") == file_name:
                                                    images_to_remove.append(image_id)
                                            
                                            for image_id in images_to_remove:
                                                cache_path = Path(image_cache.registry[image_id]["path"])
                                                if cache_path.exists():
                                                    cache_path.unlink()
                                                image_cache.current_memory -= image_cache.registry[image_id]["size"]
                                                del image_cache.registry[image_id]
                                                logger.info(f"Image cache deleted: {image_id}")
                                        
                                        # インデックスを再読み込み
                                        remaining_count = collection.count()
                                        if remaining_count > 0:
                                            # インデックスを再構築
                                            vector_store = ChromaVectorStore(chroma_collection=collection)
                                            st.session_state.index = VectorStoreIndex.from_vector_store(vector_store)
                                            st.success(f"✅ {len(deleted_files)}ファイルのインデックスを削除しました")
                                        else:
                                            # すべて削除された場合
                                            st.session_state.index = None
                                            st.session_state.index_created = False
                                            st.success("✅ すべてのインデックスが削除されました")
                                        
                                        st.balloons()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ 削除失敗: {e}")
                                        logger.error(f"Index deletion failed: {e}")
                            with col_idx2:
                                if st.button("❌ キャンセル", use_container_width=True):
                                    st.rerun()
                        else:
                            st.info("ℹ️ 削除するファイルを選択してください")
                    else:
                        st.info("ℹ️ インデックスにファイルがありません")
                except Exception as e:
                    st.error(f"❌ エラー: {e}")
                    logger.error(f"Failed to get indexed files: {e}")
        
        with idx_tab2:
            st.warning("⚠️ すべてのインデックスとチャット履歴を削除します")
            
            if st.button("🗑️ すべてのインデックスを削除", type="secondary", use_container_width=True):
                if st.session_state.get("confirm_index_delete_all", False):
                    try:
                        # ChromaDBのコレクションを削除
                        chroma_client = get_chroma_client()
                        if chroma_client:
                            try:
                                chroma_client.delete_collection("multimodal_rag")
                                logger.info("ChromaDB collection deleted")
                            except Exception as e:
                                logger.warning(f"Collection deletion warning: {e}")
                        
                        # 画像キャッシュをクリア
                        st.session_state.image_cache.clear()
                        
                        # セッション状態をリセット
                        st.session_state.index = None
                        st.session_state.index_created = False
                        st.session_state.messages = []
                        
                        # キャッシュをクリア
                        st.cache_resource.clear()
                        
                        st.success("✅ すべてのインデックスを削除しました")
                        logger.info("All indexes deleted successfully")
                        st.session_state.confirm_index_delete_all = False
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 削除失敗: {e}")
                        logger.error(f"Index deletion failed: {e}")
                        st.session_state.confirm_index_delete_all = False
                else:
                    st.session_state.confirm_index_delete_all = True
                    st.warning("⚠️ もう一度クリックして確認してください")
                    st.info("📝 注意: すべてのインデックスと画像キャッシュ、チャット履歴が削除されます")
    else:
        st.info("ℹ️ 削除可能なインデックスがありません")

# タブ2: 質問応答（マルチモーダル専用）
with tab2:
    st.header("💬 質問応答")
    
    if not st.session_state.index_created:
        st.warning("⚠️ まずインデックスを作成してください")
    else:
        chat_container = st.container(height=500)
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if (message["role"] == "assistant" and 
                        message.get("is_multimodal", False) and 
                        "image_documents" in message):
                        render_response_with_images(message["content"], message["image_documents"])
                    else:
                        st.markdown(message["content"])
                    
                    if show_sources and "sources" in message and message["sources"]:
                        with st.expander("📚 参照元"):
                            for i, source in enumerate(message["sources"]):
                                st.markdown(f"**ソース {i+1}** - {source.get('file_name', 'Unknown')} (Page {source.get('page', '?')})")
                                st.markdown(f"関連度: {source['score']:.3f}")
                                st.text(source["text"][:200] + "...")
                                st.divider()
        
        if prompt := st.chat_input("質問を入力してください..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("● 回答を生成中...")
                    
                    result = query_with_multimodal(
                        st.session_state.index,
                        prompt,
                        similarity_top_k=similarity_top_k,
                        max_tokens=st.session_state.get("max_tokens", 2000),
                        image_detail=st.session_state.get("image_detail", "high"),
                        max_images=st.session_state.get("max_images", 5),
                        response_mode=st.session_state.get("response_mode", "compact"),
                        use_chat_history=st.session_state.get("use_chat_history", True),
                        chat_history_length=st.session_state.get("chat_history_length", 5),
                        top_p=st.session_state.get("top_p", 1.0),
                        frequency_penalty=st.session_state.get("frequency_penalty", 0.0),
                        presence_penalty=st.session_state.get("presence_penalty", 0.0),
                        seed=st.session_state.get("seed", None)
                    )
                    
                    if result["success"]:
                        message_placeholder.empty()
                        if result["image_documents"]:
                            render_response_with_images(result["answer"], result["image_documents"])
                        else:
                            st.markdown(result["answer"])
                        
                        sources = []
                        for node in result["source_nodes"]:
                            sources.append({
                                "text": node.text,
                                "score": node.score,
                                "file_name": node.metadata.get("file_name", "Unknown"),
                                "page": node.metadata.get("page", "?")
                            })
                        
                        if show_sources and sources:
                            with st.expander("📚 参照元"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**ソース {i+1}** - {source.get('file_name', 'Unknown')} (Page {source.get('page', '?')})")
                                    st.markdown(f"関連度: {source['score']:.3f}")
                                    st.text(source["text"][:200] + "...")
                                    st.divider()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": sources,
                            "image_documents": result["image_documents"],
                            "is_multimodal": True
                        })
                        
                        logger.info(f"Query answered: {prompt[:50]}...")
                    else:
                        message_placeholder.markdown(f"❌ {result['answer']}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ {result['answer']}",
                            "sources": [],
                            "is_multimodal": False
                        })
                        logger.error(f"Query failed: {result['answer']}")
            
            st.rerun()

# タブ3: VectorDBブラウザー
with tab3:
    chroma_client = get_chroma_client()
    if chroma_client:
        documents = get_all_documents_from_vectordb(chroma_client)
        
        if not documents:
            st.warning("⚠️ VectorDBにドキュメントが登録されていません")
            st.info("「📚 ドキュメント管理」タブでインデックスを作成してください")
        else:
            render_vectordb_browser(chroma_client, st.session_state.image_cache)
            
            st.markdown("---")
            export_vectordb_summary(documents)

# タブ4: システム情報
with tab4:
    st.header("📊 システム情報")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 ストレージ")
        data_dir = Path("./uploaded_data")
        if data_dir.exists():
            total_size = sum(f.stat().st_size for f in data_dir.glob("*.*"))
            st.metric("データサイズ", f"{total_size / (1024*1024):.1f} MB")
        
        chroma_dir = Path("./chroma_db")
        if chroma_dir.exists():
            chroma_size = sum(f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file())
            st.metric("データベースサイズ", f"{chroma_size / (1024*1024):.1f} MB")
        
        cache_dir = Path("./image_cache")
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.glob("*") if f.is_file())
            st.metric("画像キャッシュサイズ", f"{cache_size / (1024*1024):.1f} MB")
    
    with col2:
        st.subheader("📈 セッション状態")
        st.metric("チャット履歴", len(st.session_state.messages))
        st.metric("キャッシュ画像数", len(st.session_state.image_cache.registry))
        
        if st.session_state.index_created:
            st.success("インデックス: 作成済み")
        else:
            st.info("インデックス: 未作成")
    
    st.markdown("---")
    
    st.subheader("📝 最新ログ")
    log_dir = Path("./logs")
    if log_dir.exists():
        log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            latest_log = log_files[0]
            with open(latest_log, 'r', encoding='utf-8') as f:
                log_content = f.readlines()
            
            st.text(f"ファイル: {latest_log.name}")
            st.code("".join(log_content[-20:]), language="log")
        else:
            st.info("ログファイルがありません")
    else:
        st.info("ログディレクトリがありません")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    🔍 マルチモーダルRAGシステム v3.0 | 🤖 GPT-4 Vision専用 | 🖼️ 画像理解 | 🔍 VectorDBブラウザー
    </div>
    """,
    unsafe_allow_html=True
)