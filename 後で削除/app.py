import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb
import os
from pathlib import Path
from dotenv import load_dotenv
import shutil
import pypdf
import pdfplumber
from PIL import Image
import io
import fitz
import base64
import json


# 環境変数の読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="マルチモーダルRAGシステム",
    page_icon="🔍",
    layout="wide"
)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_created" not in st.session_state:
    st.session_state.index_created = False
if "pdf_images" not in st.session_state:
    st.session_state.pdf_images = {}
if "image_registry" not in st.session_state:
    st.session_state.image_registry = {}  # image_id -> image_data
if "use_multimodal" not in st.session_state:
    st.session_state.use_multimodal = False

@st.cache_resource
def get_chroma_client():
    """ChromaDBクライアントの取得"""
    client = chromadb.PersistentClient(path="./chroma_db")
    return client

@st.cache_resource
def initialize_rag_system(_chroma_client, collection_name="multimodal_rag"):
    """RAGシステムの初期化"""
    try:
        _chroma_client.delete_collection(collection_name)
    except:
        pass
    
    chroma_collection = _chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 埋め込みモデルとLLMの設定
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    
    return storage_context

def extract_images_high_quality(pdf_path, dpi=300):
    """ページ全体を高品質画像化"""
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            images.append({
                "page": page_num + 1,
                "image": image,
                "type": "full_page",
                "file_name": pdf_path.name
            })
        pdf_document.close()
    except Exception as e:
        st.warning(f"ページ画像化エラー: {str(e)}")
    return images

def extract_images_embedded_positioned(pdf_path, min_size=100):
    """🌟 位置情報ベースで画像を正確に切り抜く"""
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    
                    if not rects:
                        continue
                    
                    for rect_index, rect in enumerate(rects):
                        x0, y0, x1, y1 = rect
                        width = abs(x1 - x0)
                        height = abs(y1 - y0)
                        
                        if width < min_size or height < min_size:
                            continue
                        
                        aspect_ratio = width / height if height > 0 else 0
                        if aspect_ratio > 10 or aspect_ratio < 0.1:
                            continue
                        
                        mat = fitz.Matrix(2.0, 2.0)
                        clip_rect = fitz.Rect(x0, y0, x1, y1)
                        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        if image.width < min_size or image.height < min_size:
                            continue
                        
                        images.append({
                            "page": page_num + 1,
                            "image": image,
                            "type": "embedded",
                            "file_name": pdf_path.name,
                            "index": img_index + 1,
                            "rect_index": rect_index + 1
                        })
                except:
                    continue
        pdf_document.close()
    except Exception as e:
        st.warning(f"埋め込み画像抽出エラー: {str(e)}")
    return images

def extract_images_from_pdf(pdf_path, method="high_quality", dpi=300):
    """画像抽出の統合関数"""
    if method == "high_quality":
        return extract_images_high_quality(pdf_path, dpi=300)
    elif method == "medium_quality":
        return extract_images_high_quality(pdf_path, dpi=150)
    elif method == "embedded":
        return extract_images_embedded_positioned(pdf_path)
    elif method == "combined":
        page_images = extract_images_high_quality(pdf_path, dpi=200)
        embedded_images = extract_images_embedded_positioned(pdf_path)
        return page_images + embedded_images
    else:
        return extract_images_high_quality(pdf_path, dpi=300)

def extract_text_from_pdf(pdf_path):
    """PDFからページ単位でテキストを抽出"""
    page_texts = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    page_texts[page_num] = page_text
        
        if page_texts:
            return page_texts
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    page_texts[page_num] = page_text
        
        return page_texts
    except Exception as e:
        st.warning(f"PDF読み込みエラー: {str(e)}")
        return {}

def register_images(images, file_name):
    """
    🆕 画像レジストリに登録してIDを割り当て
    """
    image_ids_by_page = {}
    
    for img_data in images:
        page_num = img_data["page"]
        
        # ユニークなimage_idを生成
        image_id = f"{file_name}_p{page_num}_t{img_data['type']}"
        if "index" in img_data:
            image_id += f"_i{img_data['index']}"
        if "rect_index" in img_data:
            image_id += f"_r{img_data['rect_index']}"
        
        # レジストリに登録
        st.session_state.image_registry[image_id] = img_data
        
        # ページごとのimage_idリストを作成
        if page_num not in image_ids_by_page:
            image_ids_by_page[page_num] = []
        image_ids_by_page[page_num].append(image_id)
    
    return image_ids_by_page

def load_and_index_documents(data_dir, storage_context, extraction_method, dpi):
    """
    🌟 改善版: ページ単位でドキュメントを分割
    """
    try:
        data_path = Path(data_dir)
        all_files = list(data_path.glob("*.*"))
        
        st.info(f"📁 検出されたファイル: {len(all_files)}件")
        
        method_names = {
            "high_quality": f"高品質ページ全体（DPI {dpi}）",
            "medium_quality": f"中品質ページ全体（DPI {dpi}）",
            "embedded": "🌟 埋め込み画像（位置情報ベース）",
            "combined": "ページ全体+埋め込み画像"
        }
        st.info(f"🎨 画像抽出方法: {method_names.get(extraction_method, extraction_method)}")
        
        documents = []
        
        for file_path in all_files:
            if file_path.suffix.lower() == '.pdf':
                st.info(f"📄 PDFを処理中: {file_path.name}")
                
                # ページ単位でテキスト抽出
                page_texts = extract_text_from_pdf(file_path)
                
                # 画像抽出
                with st.spinner(f"画像を抽出中..."):
                    images = extract_images_from_pdf(file_path, method=extraction_method, dpi=dpi)
                
                if images:
                    st.session_state.pdf_images[file_path.name] = images
                    
                    # 🆕 画像をレジストリに登録
                    image_ids_by_page = register_images(images, file_path.name)
                    
                    st.success(f"🖼️ {len(images)}枚の画像を抽出しました")
                else:
                    image_ids_by_page = {}
                
                if page_texts:
                    # 🌟 ページ単位でDocumentを作成（重要！）
                    for page_num, page_text in page_texts.items():
                        # このページの画像IDリストを取得
                        page_image_ids = image_ids_by_page.get(page_num, [])
                        
                        # 🔧 リストをJSON文字列に変換（ChromaDB対応）
                        image_ids_json = json.dumps(page_image_ids)
                        
                        doc = Document(
                            text=page_text,
                            metadata={
                                "file_name": file_path.name,
                                "page": page_num,
                                "total_pages": len(page_texts),
                                "image_ids": image_ids_json,  # 🆕 JSON文字列として保存
                                "num_images": len(page_image_ids)
                            }
                        )
                        documents.append(doc)
                    
                    st.success(f"✅ {len(page_texts)}ページをインデックス化")
                    
                    # プレビュー表示
                    with st.expander(f"📄 {file_path.name} のプレビュー"):
                        st.text(f"総ページ数: {len(page_texts)}")
                        st.text(f"総画像数: {len(images)}")
                        
                        if images:
                            st.markdown("---")
                            st.markdown("**抽出された画像（最初の6枚）:**")
                            cols = st.columns(3)
                            for idx, img_data in enumerate(images[:6]):
                                with cols[idx % 3]:
                                    caption = f"Page {img_data['page']} ({img_data['type']})"
                                    st.image(img_data["image"], caption=caption, use_container_width=True)
            
            elif file_path.suffix.lower() in ['.txt', '.md']:
                try:
                    encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp']
                    text = None
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                text = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if text:
                        doc = Document(
                            text=text,
                            metadata={
                                "file_name": file_path.name,
                                "file_type": file_path.suffix[1:]
                            }
                        )
                        documents.append(doc)
                except Exception as e:
                    st.warning(f"⚠️ {file_path.name}: {str(e)}")
        
        if not documents:
            return None, "ドキュメントを読み込めませんでした"
        
        st.success(f"✅ {len(documents)}個のドキュメントチャンクを作成")
        
        with st.spinner("ベクトルインデックスを作成中..."):
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
        
        return index, None
        
    except Exception as e:
        import traceback
        return None, f"エラー: {str(e)}\n\n{traceback.format_exc()}"

def get_images_from_node(node):
    """
    🆕 Nodeのメタデータから画像を取得
    """
    images = []
    if hasattr(node, 'metadata') and 'image_ids' in node.metadata:
        # JSON文字列をパース
        try:
            image_ids_str = node.metadata['image_ids']
            if isinstance(image_ids_str, str):
                image_ids = json.loads(image_ids_str)
            else:
                image_ids = image_ids_str  # 既にリストの場合
            
            for image_id in image_ids:
                if image_id in st.session_state.image_registry:
                    images.append(st.session_state.image_registry[image_id])
        except (json.JSONDecodeError, TypeError) as e:
            st.warning(f"画像ID解析エラー: {e}")
    return images

def create_multimodal_prompt(query, context_text, context_images):
    """
    🆕 マルチモーダルLLM用のプロンプトを作成
    """
    prompt = f"""以下のコンテキストに基づいて質問に答えてください。

【テキスト情報】
{context_text}

【質問】
{query}

【回答】
"""
    return prompt

# メインUI
st.title("🔍 マルチモーダルRAGシステム（改善版）")
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
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.success("✅ APIキーが設定されました")
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
    
    st.markdown("---")
    
    # マルチモーダル設定
    st.subheader("🤖 LLM設定")
    use_multimodal = st.checkbox(
        "マルチモーダルLLM使用（画像理解）",
        value=False,
        help="GPT-4 Visionで画像も理解（開発中）"
    )
    st.session_state.use_multimodal = use_multimodal
    
    if use_multimodal:
        st.info("🔬 実験的機能: 画像+テキストを同時分析")
    
    st.markdown("---")
    
    # 統計情報
    st.subheader("📊 統計情報")
    data_dir = Path("./uploaded_data")
    if data_dir.exists():
        files = list(data_dir.glob("*.*"))
        st.metric("ファイル", len(files))
    else:
        st.metric("ファイル", 0)
    
    total_images = len(st.session_state.image_registry)
    if total_images > 0:
        st.metric("画像", total_images)
    
    if st.session_state.index_created:
        st.success("✅ インデックス作成済み")
    else:
        st.info("ℹ️ インデックス未作成")
    
    st.markdown("---")
    
    # 表示設定
    st.subheader("👁️ 表示設定")
    show_images_in_chat = st.checkbox("チャットに画像を表示", value=True)
    
    st.markdown("---")
    
    # リセット
    if st.button("🗑️ 全データをリセット", type="secondary"):
        if st.session_state.get("confirm_reset", False):
            if data_dir.exists():
                shutil.rmtree(data_dir)
                data_dir.mkdir()
            
            chroma_dir = Path("./chroma_db")
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
                chroma_dir.mkdir()
            
            st.session_state.index_created = False
            st.session_state.messages = []
            st.session_state.pdf_images = {}
            st.session_state.image_registry = {}
            st.cache_resource.clear()
            st.success("✅ リセット完了")
            st.session_state.confirm_reset = False
            st.rerun()
        else:
            st.session_state.confirm_reset = True
            st.warning("⚠️ もう一度クリックして確認")

# メインコンテンツ
if not api_key_input:
    st.info("👈 サイドバーからOpenAI APIキーを入力してください")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📚 ドキュメント管理", "💬 質問応答", "🖼️ 画像ギャラリー"])

with tab1:
    st.header("📚 ドキュメント管理")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "ファイルをアップロード",
            accept_multiple_files=True,
            type=["txt", "pdf", "md"]
        )
        
        if uploaded_files:
            data_dir = Path("./uploaded_data")
            data_dir.mkdir(exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = data_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ {len(uploaded_files)}件保存")
    
    with col2:
        st.subheader("ファイル一覧")
        data_dir = Path("./uploaded_data")
        if data_dir.exists():
            files = list(data_dir.glob("*.*"))
            for file in files:
                size_kb = file.stat().st_size / 1024
                st.text(f"📄 {file.name} ({size_kb:.1f}KB)")
    
    st.markdown("---")
    
    if st.button("🔨 インデックスを作成", type="primary", use_container_width=True):
        data_dir = Path("./uploaded_data")
        
        if not data_dir.exists() or not list(data_dir.glob("*.*")):
            st.error("❌ ファイルがありません")
        else:
            with st.spinner("インデックスを作成中..."):
                try:
                    chroma_client = get_chroma_client()
                    storage_context = initialize_rag_system(chroma_client)
                    
                    index, error = load_and_index_documents(
                        str(data_dir), 
                        storage_context,
                        extraction_method,
                        dpi
                    )
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.session_state.index = index
                        st.session_state.index_created = True
                        st.success("✅ インデックス作成完了！")
                        st.balloons()
                except Exception as e:
                    import traceback
                    st.error(f"❌ {str(e)}")
                    with st.expander("詳細"):
                        st.code(traceback.format_exc())

with tab2:
    st.header("💬 質問応答")
    
    if not st.session_state.index_created:
        st.warning("⚠️ まずインデックスを作成してください")
    else:
        # チャット履歴
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "images" in message and message["images"] and show_images_in_chat:
                    st.markdown("---")
                    st.markdown("**📸 関連画像:**")
                    cols = st.columns(min(3, len(message["images"])))
                    for idx, img_data in enumerate(message["images"]):
                        with cols[idx % 3]:
                            caption = f"{img_data['file_name']} - Page {img_data['page']}"
                            st.image(img_data["image"], caption=caption, use_container_width=True)
                
                if "sources" in message and message["sources"]:
                    with st.expander("📚 参照元"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**ソース {i+1}** - {source.get('file_name', 'Unknown')} (Page {source.get('page', '?')})")
                            st.markdown(f"関連度: {source['score']:.3f}")
                            st.text(source["text"][:200] + "...")
                            st.divider()
        
        # 質問入力
        if prompt := st.chat_input("質問を入力..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("回答生成中..."):
                    try:
                        query_engine = st.session_state.index.as_query_engine(
                            similarity_top_k=3,
                            response_mode="compact"
                        )
                        
                        response = query_engine.query(prompt)
                        st.markdown(response.response)
                        
                        # 🆕 Nodeから直接画像を取得
                        sources = []
                        all_images = []
                        
                        for node in response.source_nodes:
                            sources.append({
                                "text": node.text,
                                "score": node.score,
                                "file_name": node.metadata.get("file_name", "Unknown"),
                                "page": node.metadata.get("page", "?")
                            })
                            
                            # メタデータから画像を取得
                            node_images = get_images_from_node(node)
                            all_images.extend(node_images)
                        
                        # 重複削除
                        seen = set()
                        unique_images = []
                        for img in all_images:
                            key = (img['file_name'], img['page'], img['type'])
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
                                    caption = f"{img_data['file_name']} - Page {img_data['page']}"
                                    st.image(img_data["image"], caption=caption, use_container_width=True)
                        
                        # 参照元表示
                        if sources:
                            with st.expander("📚 参照元"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**ソース {i+1}** - {source['file_name']} (Page {source['page']})")
                                    st.markdown(f"関連度: {source['score']:.3f}")
                                    st.text(source["text"][:200] + "...")
                                    st.divider()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.response,
                            "sources": sources,
                            "images": unique_images
                        })
                    except Exception as e:
                        error_msg = f"❌ エラー: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

with tab3:
    st.header("🖼️ 画像ギャラリー")
    
    if not st.session_state.pdf_images:
        st.info("📄 画像がここに表示されます")
    else:
        for pdf_name, images in st.session_state.pdf_images.items():
            with st.expander(f"📄 {pdf_name} ({len(images)}枚)", expanded=True):
                # ページごとにグループ化
                pages = {}
                for img in images:
                    page = img["page"]
                    if page not in pages:
                        pages[page] = []
                    pages[page].append(img)
                
                for page_num in sorted(pages.keys()):
                    st.markdown(f"**Page {page_num}**")
                    page_images = pages[page_num]
                    cols = st.columns(3)
                    for idx, img_data in enumerate(page_images):
                        with cols[idx % 3]:
                            st.image(img_data["image"], use_container_width=True)
                            
                            img_bytes = io.BytesIO()
                            img_data["image"].save(img_bytes, format="PNG")
                            st.download_button(
                                "💾",
                                img_bytes.getvalue(),
                                f"{pdf_name}_p{page_num}_{idx}.png",
                                "image/png",
                                key=f"dl_{pdf_name}_{page_num}_{idx}"
                            )
                    st.markdown("---")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    🔍 マルチモーダルRAGシステム v2.0 | ページ分割・画像紐付け・マルチモーダル対応
    </div>
    """,
    unsafe_allow_html=True
)