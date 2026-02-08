"""
メインアプリケーション - 改善版 v2.2
🆕 VectorDBブラウザー追加
"""
import streamlit as st
import chromadb
import os
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv

# カスタムモジュール
from core.rag_engine import initialize_rag_system, load_and_index_documents, query_index
from core.image_handler import ImageCache
from core.multimodal_query import query_with_multimodal, render_response_with_images
from core.vectordb_browser import render_vectordb_browser, export_vectordb_summary, get_all_documents_from_vectordb
from utils.logger import get_logger
from utils.exceptions import (
    APIKeyError, FileUploadError, IndexCreationError, 
    QueryError, PDFProcessingError
)

# 環境変数の読み込み
load_dotenv()

# ロガー初期化
logger = get_logger()
logger.info("=" * 50)
logger.info("Application started - VectorDB Browser v2.2")
logger.info("=" * 50)

# ページ設定
st.set_page_config(
    page_title="マルチモーダルRAGシステム v2.2",
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
if "use_multimodal" not in st.session_state:
    st.session_state.use_multimodal = True


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


def get_images_from_node(node):
    """Nodeのメタデータから画像を取得"""
    images = []
    image_cache = st.session_state.image_cache
    
    try:
        if hasattr(node, 'metadata') and 'image_ids' in node.metadata:
            image_ids_str = node.metadata['image_ids']
            
            if isinstance(image_ids_str, str):
                image_ids = json.loads(image_ids_str)
            else:
                image_ids = image_ids_str
            
            for image_id in image_ids:
                cached_data = image_cache.get_image(image_id)
                if cached_data:
                    images.append({
                        **cached_data["metadata"],
                        "image": cached_data["image"]
                    })
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse image IDs: {e}")
    except Exception as e:
        logger.error(f"Error getting images from node: {e}")
    
    return images


# メインUI
st.title("🔍 マルチモーダルRAGシステム v2.2")
st.caption("🆕 VectorDBブラウザー追加 | 文章中に画像埋め込み | エラーハンドリング強化")
st.markdown("---")

# サイドバー（省略 - 前と同じ）
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
        except APIKeyError as e:
            st.error(f"❌ {str(e)}")
    else:
        st.warning("⚠️ APIキーを入力してください")
    
    st.markdown("---")
    
    st.subheader("🤖 回答モード")
    use_multimodal = st.checkbox(
        "マルチモーダルモード",
        value=st.session_state.use_multimodal
    )
    st.session_state.use_multimodal = use_multimodal
    
    st.markdown("---")
    
    st.subheader("🎨 画像抽出設定")
    extraction_method = st.selectbox(
        "抽出方法",
        options=["high_quality", "medium_quality", "embedded", "combined"],
        format_func=lambda x: {
            "high_quality": "高品質", "medium_quality": "中品質",
            "embedded": "埋め込み", "combined": "全て"
        }[x]
    )
    
    if extraction_method in ["high_quality", "medium_quality", "combined"]:
        dpi = st.slider("DPI", 72, 300, 200, 50)
    else:
        dpi = 150
    
    max_workers = st.slider("並列処理", 1, 5, 3)
    
    st.markdown("---")
    
    st.subheader("🔍 検索設定")
    similarity_top_k = st.slider("検索結果件数", 1, 10, 3)
    
    st.markdown("---")
    
    st.subheader("📊 統計")
    data_dir = Path("./uploaded_data")
    if data_dir.exists():
        files = list(data_dir.glob("*.*"))
        st.metric("ファイル", len(files))
    
    total_images = len(st.session_state.image_cache.registry)
    if total_images > 0:
        st.metric("画像", total_images)
    
    if st.session_state.index_created:
        st.success("✅ インデックス作成済み")
    else:
        st.info("ℹ️ インデックス未作成")
    
    st.markdown("---")
    
    show_sources = st.checkbox("参照元を表示", value=True)

# メインコンテンツ
if not api_key_input:
    st.info("👈 サイドバーからOpenAI APIキーを入力してください")
    st.stop()

# 🆕 タブを4つに増やす
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 ドキュメント管理", 
    "💬 質問応答", 
    "🔍 VectorDBブラウザー",  # 🆕 新しいタブ
    "📊 システム情報"
])

# タブ1: ドキュメント管理（省略 - app_multimodal.pyと同じ）
with tab1:
    st.header("📚 ドキュメント管理")
    # ... 省略（前と同じ）

# タブ2: 質問応答（省略 - app_multimodal.pyと同じ）
with tab2:
    st.header("💬 質問応答")
    # ... 省略（前と同じ）

# 🆕 タブ3: VectorDBブラウザー
with tab3:
    if not st.session_state.index_created:
        st.warning("⚠️ まずインデックスを作成してください")
    else:
        chroma_client = get_chroma_client()
        if chroma_client:
            render_vectordb_browser(chroma_client, st.session_state.image_cache)
            
            # エクスポート機能
            st.markdown("---")
            documents = get_all_documents_from_vectordb(chroma_client)
            if documents:
                export_vectordb_summary(documents)

# タブ4: システム情報（省略 - app_multimodal.pyと同じ）
with tab4:
    st.header("📊 システム情報")
    # ... 省略（前と同じ）

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    🔍 マルチモーダルRAGシステム v2.2 | 🆕 VectorDBブラウザー | 文章中に画像埋め込み
    </div>
    """,
    unsafe_allow_html=True
)
