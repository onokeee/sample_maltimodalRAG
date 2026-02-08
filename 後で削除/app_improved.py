"""
メインアプリケーション - 改善版
優先度:高の改善を全て反映
- エラーハンドリング強化
- 並列処理によるパフォーマンス最適化
- メモリ管理改善
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
logger.info("Application started")
logger.info("=" * 50)

# ページ設定
st.set_page_config(
    page_title="マルチモーダルRAGシステム v2.0",
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
    st.session_state.use_multimodal = False


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
    # ファイルサイズ制限（100MB）
    max_size_mb = 100
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        raise FileUploadError(
            f"ファイルサイズが大きすぎます: {file_size_mb:.1f}MB（上限: {max_size_mb}MB）"
        )
    
    # ファイルタイプ検証
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
st.title("🔍 マルチモーダルRAGシステム v2.0")
st.caption("エラーハンドリング強化 | 並列処理 | メモリ最適化")
st.markdown("---")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    # APIキー設定
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
    
    # 画像抽出設定
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
    
    # 並列処理設定
    max_workers = st.slider(
        "並列処理スレッド数",
        min_value=1,
        max_value=5,
        value=3,
        help="PDFファイルの処理を並列化します（速度向上）"
    )
    
    st.markdown("---")
    
    # 検索設定
    st.subheader("🔍 検索設定")
    similarity_top_k = st.slider(
        "検索結果件数",
        min_value=1,
        max_value=10,
        value=3,
        help="類似度が高い上位N件を取得"
    )
    
    st.markdown("---")
    
    # 統計情報
    st.subheader("📊 統計情報")
    data_dir = Path("./uploaded_data")
    if data_dir.exists():
        files = list(data_dir.glob("*.*"))
        st.metric("ファイル", len(files))
    else:
        st.metric("ファイル", 0)
    
    # キャッシュ情報
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
    
    # 表示設定
    st.subheader("👁️ 表示設定")
    show_images_in_chat = st.checkbox("チャットに画像を表示", value=True)
    show_sources = st.checkbox("参照元を表示", value=True)
    
    st.markdown("---")
    
    # リセット
    if st.button("🗑️ 全データをリセット", type="secondary"):
        if st.session_state.get("confirm_reset", False):
            try:
                # ファイル削除
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                    data_dir.mkdir()
                
                # DB削除
                chroma_dir = Path("./chroma_db")
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir)
                    chroma_dir.mkdir()
                
                # キャッシュクリア
                st.session_state.image_cache.clear()
                
                # セッション状態リセット
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

tab1, tab2, tab3 = st.tabs(["📚 ドキュメント管理", "💬 質問応答", "📊 システム情報"])

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
                    # ファイル検証
                    validate_file_upload(uploaded_file)
                    
                    # 保存
                    file_path = data_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    success_count += 1
                    logger.info(f"File uploaded: {uploaded_file.name}")
                
                except FileUploadError as e:
                    st.error(f"❌ {uploaded_file.name}: {str(e)}")
                    error_count += 1
                    logger.warning(f"File upload failed: {uploaded_file.name} - {e}")
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name}: 予期しないエラー - {str(e)}")
                    error_count += 1
                    logger.error(f"Unexpected error during upload: {e}")
            
            if success_count > 0:
                st.success(f"✅ {success_count}件保存完了")
            if error_count > 0:
                st.warning(f"⚠️ {error_count}件失敗")
    
    with col2:
        st.subheader("📁 ファイル一覧")
        data_dir = Path("./uploaded_data")
        if data_dir.exists():
            files = sorted(data_dir.glob("*.*"))
            if files:
                for file in files:
                    size_kb = file.stat().st_size / 1024
                    icon = "📄" if file.suffix.lower() == ".pdf" else "📝"
                    st.text(f"{icon} {file.name} ({size_kb:.1f}KB)")
            else:
                st.info("ファイルなし")
        else:
            st.info("ファイルなし")
    
    st.markdown("---")
    
    # インデックス作成
    if st.button("🔨 インデックスを作成", type="primary", use_container_width=True):
        data_dir = Path("./uploaded_data")
        
        if not data_dir.exists() or not list(data_dir.glob("*.*")):
            st.error("❌ アップロードされたファイルがありません")
            logger.warning("No files to index")
        else:
            try:
                with st.spinner("インデックスを作成中..."):
                    chroma_client = get_chroma_client()
                    
                    if chroma_client is None:
                        raise IndexCreationError("データベースクライアントの初期化に失敗しました")
                    
                    storage_context = initialize_rag_system(chroma_client)
                    
                    index, error = load_and_index_documents(
                        str(data_dir),
                        storage_context,
                        extraction_method,
                        dpi,
                        max_workers=max_workers
                    )
                    
                    if error:
                        raise IndexCreationError(error)
                    
                    st.session_state.index = index
                    st.session_state.index_created = True
                    st.success("✅ インデックス作成完了！")
                    st.balloons()
                    logger.info("Index created successfully")
            
            except IndexCreationError as e:
                st.error(f"❌ インデックス作成エラー: {str(e)}")
                logger.error(f"Index creation failed: {e}")
            except PDFProcessingError as e:
                st.error(f"❌ PDF処理エラー: {str(e)}")
                logger.error(f"PDF processing failed: {e}")
            except Exception as e:
                st.error(f"❌ 予期しないエラー: {str(e)}")
                logger.error(f"Unexpected error during indexing: {e}", exc_info=True)
                with st.expander("詳細エラー情報"):
                    st.code(str(e))

with tab2:
    st.header("💬 質問応答")
    
    if not st.session_state.index_created:
        st.warning("⚠️ まずインデックスを作成してください")
    else:
        # チャット履歴表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 画像表示
                if (message["role"] == "assistant" and 
                    "images" in message and 
                    message["images"] and 
                    show_images_in_chat):
                    st.markdown("---")
                    st.markdown("**📸 関連画像:**")
                    cols = st.columns(min(3, len(message["images"])))
                    for idx, img_data in enumerate(message["images"][:6]):
                        with cols[idx % 3]:
                            caption = f"{img_data.get('file_name', 'Unknown')} - Page {img_data.get('page', '?')}"
                            st.image(img_data["image"], caption=caption, use_container_width=True)
                
                # 参照元表示
                if show_sources and "sources" in message and message["sources"]:
                    with st.expander("📚 参照元"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**ソース {i+1}** - {source.get('file_name', 'Unknown')} (Page {source.get('page', '?')})")
                            st.markdown(f"関連度: {source['score']:.3f}")
                            st.text(source["text"][:200] + "...")
                            st.divider()
        
        # 質問入力
        if prompt := st.chat_input("質問を入力してください..."):
            # ユーザーメッセージ表示
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # アシスタント応答
            with st.chat_message("assistant"):
                try:
                    with st.spinner("回答生成中..."):
                        response = query_index(
                            st.session_state.index,
                            prompt,
                            similarity_top_k=similarity_top_k
                        )
                        
                        st.markdown(response.response)
                        
                        # ソース情報収集
                        sources = []
                        all_images = []
                        
                        for node in response.source_nodes:
                            sources.append({
                                "text": node.text,
                                "score": node.score,
                                "file_name": node.metadata.get("file_name", "Unknown"),
                                "page": node.metadata.get("page", "?")
                            })
                            
                            # 画像取得
                            node_images = get_images_from_node(node)
                            all_images.extend(node_images)
                        
                        # 重複削除
                        seen = set()
                        unique_images = []
                        for img in all_images:
                            key = (img.get('file_name'), img.get('page'), img.get('type'))
                            if key not in seen:
                                seen.add(key)
                                unique_images.append(img)
                        
                        # 画像表示
                        if unique_images and show_images_in_chat:
                            st.markdown("---")
                            st.markdown("**📸 関連画像:**")
                            cols = st.columns(min(3, len(unique_images)))
                            for idx, img_data in enumerate(unique_images[:6]):
                                with cols[idx % 3]:
                                    caption = f"{img_data.get('file_name', 'Unknown')} - Page {img_data.get('page', '?')}"
                                    st.image(img_data["image"], caption=caption, use_container_width=True)
                        
                        # 参照元表示
                        if show_sources and sources:
                            with st.expander("📚 参照元"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**ソース {i+1}** - {source['file_name']} (Page {source['page']})")
                                    st.markdown(f"関連度: {source['score']:.3f}")
                                    st.text(source["text"][:200] + "...")
                                    st.divider()
                        
                        # メッセージ保存
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.response,
                            "sources": sources,
                            "images": unique_images
                        })
                        
                        logger.info(f"Query answered: {prompt[:50]}...")
                
                except QueryError as e:
                    error_msg = f"❌ 検索エラー: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Query error: {e}")
                except Exception as e:
                    error_msg = f"❌ 予期しないエラー: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Unexpected error during query: {e}", exc_info=True)

with tab3:
    st.header("📊 システム情報")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 ストレージ")
        
        # データディレクトリ
        data_dir = Path("./uploaded_data")
        if data_dir.exists():
            total_size = sum(f.stat().st_size for f in data_dir.glob("*.*"))
            st.metric("データサイズ", f"{total_size / (1024*1024):.1f} MB")
        
        # ChromaDB
        chroma_dir = Path("./chroma_db")
        if chroma_dir.exists():
            chroma_size = sum(f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file())
            st.metric("データベースサイズ", f"{chroma_size / (1024*1024):.1f} MB")
        
        # 画像キャッシュ
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
    
    # ログファイル表示
    st.subheader("📝 最新ログ")
    log_dir = Path("./logs")
    if log_dir.exists():
        log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            latest_log = log_files[0]
            with open(latest_log, 'r', encoding='utf-8') as f:
                log_content = f.readlines()
            
            st.text(f"ファイル: {latest_log.name}")
            st.code("".join(log_content[-20:]), language="log")  # 最新20行
        else:
            st.info("ログファイルがありません")
    else:
        st.info("ログディレクトリがありません")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    🔍 マルチモーダルRAGシステム v2.0 | エラーハンドリング強化 | 並列処理 | メモリ最適化
    </div>
    """,
    unsafe_allow_html=True
)
