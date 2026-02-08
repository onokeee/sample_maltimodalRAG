"""
RAGエンジンモジュール - 並列処理対応
"""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb

from core.pdf_processor import extract_text_from_pdf, validate_pdf_file
from core.image_handler import extract_images_from_pdf, ImageCache
from utils.logger import get_logger
from utils.exceptions import IndexCreationError, QueryError

logger = get_logger()


def initialize_image_cache():
    """画像キャッシュの初期化"""
    if "image_cache" not in st.session_state:
        st.session_state.image_cache = ImageCache()
        logger.info("Image cache initialized")
    return st.session_state.image_cache


def initialize_rag_system(chroma_client, collection_name="multimodal_rag"):
    """RAGシステムの初期化"""
    logger.info(f"Initializing RAG system with collection: {collection_name}")
    
    try:
        # 既存コレクションを削除
        try:
            chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
        
        # 新規作成
        chroma_collection = chroma_client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 埋め込みモデルとLLMの設定（セッション状態から取得）
        embed_model = st.session_state.get("embed_model", "text-embedding-3-small")
        llm_model = st.session_state.get("llm_model", "gpt-4o-mini")
        temperature = st.session_state.get("temperature", 0.1)
        
        Settings.embed_model = OpenAIEmbedding(model=embed_model)
        Settings.llm = OpenAI(model=llm_model, temperature=temperature)
        
        logger.info(f"RAG system settings: embed={embed_model}, llm={llm_model}, temp={temperature}")
        
        logger.info("RAG system initialized successfully")
        return storage_context
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise IndexCreationError(f"RAGシステムの初期化に失敗しました: {e}")


def process_single_pdf(file_path, extraction_method, dpi, image_cache):
    """単一PDFファイルの処理（並列処理用）"""
    logger.info(f"Processing PDF: {file_path.name}")
    documents = []
    
    try:
        # PDFバリデーション
        validate_pdf_file(file_path)
        
        # テキスト抽出
        page_texts = extract_text_from_pdf(file_path)
        
        # 画像抽出
        images = extract_images_from_pdf(file_path, method=extraction_method, dpi=dpi)
        
        # 画像をキャッシュに登録
        image_ids_by_page = {}
        for img_data in images:
            page_num = img_data["page"]
            
            # ユニークIDを生成
            image_id = f"{file_path.name}_p{page_num}_t{img_data['type']}"
            if "index" in img_data:
                image_id += f"_i{img_data['index']}"
            if "rect_index" in img_data:
                image_id += f"_r{img_data['rect_index']}"
            
            # キャッシュに追加
            metadata = {
                "file_name": img_data["file_name"],
                "page": page_num,
                "type": img_data["type"]
            }
            image_cache.add_image(image_id, img_data["image"], metadata)
            
            # ページごとのimage_idリストを作成
            if page_num not in image_ids_by_page:
                image_ids_by_page[page_num] = []
            image_ids_by_page[page_num].append(image_id)
        
        # ドキュメント作成
        for page_num, page_text in page_texts.items():
            page_image_ids = image_ids_by_page.get(page_num, [])
            image_ids_json = json.dumps(page_image_ids)
            
            doc = Document(
                text=page_text,
                metadata={
                    "file_name": file_path.name,
                    "page": page_num,
                    "total_pages": len(page_texts),
                    "image_ids": image_ids_json,
                    "num_images": len(page_image_ids)
                }
            )
            documents.append(doc)
        
        logger.info(f"PDF processed successfully: {file_path.name} ({len(documents)} documents, {len(images)} images)")
        return {
            "success": True,
            "file_name": file_path.name,
            "documents": documents,
            "num_images": len(images),
            "num_pages": len(page_texts)
        }
    
    except Exception as e:
        logger.error(f"Failed to process PDF {file_path.name}: {e}")
        return {
            "success": False,
            "file_name": file_path.name,
            "error": str(e)
        }


def process_text_file(file_path):
    """テキストファイルの処理"""
    logger.info(f"Processing text file: {file_path.name}")
    
    try:
        encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp']
        text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                logger.debug(f"Successfully read with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if not text:
            raise ValueError("すべてのエンコーディングで読み込みに失敗しました")
        
        doc = Document(
            text=text,
            metadata={
                "file_name": file_path.name,
                "file_type": file_path.suffix[1:]
            }
        )
        
        logger.info(f"Text file processed: {file_path.name}")
        return {
            "success": True,
            "file_name": file_path.name,
            "documents": [doc]
        }
    
    except Exception as e:
        logger.error(f"Failed to process text file {file_path.name}: {e}")
        return {
            "success": False,
            "file_name": file_path.name,
            "error": str(e)
        }


def load_and_index_documents(data_dir, storage_context, extraction_method, dpi, max_workers=3):
    """
    ドキュメントの読み込みとインデックス化（並列処理対応）
    """
    logger.info(f"Starting document loading from {data_dir}")
    
    try:
        data_path = Path(data_dir)
        all_files = list(data_path.glob("*.*"))
        
        if not all_files:
            logger.warning("No files found in directory")
            return None, "ディレクトリにファイルがありません"
        
        logger.info(f"Found {len(all_files)} files")
        
        # 画像キャッシュ初期化
        image_cache = initialize_image_cache()
        
        all_documents = []
        pdf_files = [f for f in all_files if f.suffix.lower() == '.pdf']
        text_files = [f for f in all_files if f.suffix.lower() in ['.txt', '.md']]
        
        # PDFファイルを並列処理
        if pdf_files:
            st.info(f"📄 PDFファイルを並列処理中... ({len(pdf_files)}件)")
            progress_bar = st.progress(0)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # タスクを投入
                future_to_file = {
                    executor.submit(
                        process_single_pdf, 
                        file_path, 
                        extraction_method, 
                        dpi, 
                        image_cache
                    ): file_path 
                    for file_path in pdf_files
                }
                
                # 結果を収集
                completed = 0
                for future in as_completed(future_to_file):
                    result = future.result()
                    completed += 1
                    progress_bar.progress(completed / len(pdf_files))
                    
                    if result["success"]:
                        all_documents.extend(result["documents"])
                        st.success(f"✅ {result['file_name']}: {result['num_pages']}ページ, {result['num_images']}画像")
                    else:
                        st.error(f"❌ {result['file_name']}: {result['error']}")
            
            progress_bar.empty()
        
        # テキストファイルを処理
        for file_path in text_files:
            result = process_text_file(file_path)
            if result["success"]:
                all_documents.extend(result["documents"])
                st.success(f"✅ {result['file_name']}")
            else:
                st.error(f"❌ {result['file_name']}: {result['error']}")
        
        if not all_documents:
            logger.warning("No documents were created")
            return None, "ドキュメントを読み込めませんでした"
        
        st.success(f"✅ {len(all_documents)}個のドキュメントチャンクを作成")
        
        # インデックス作成
        with st.spinner("ベクトルインデックスを作成中..."):
            logger.info("Creating vector index...")
            index = VectorStoreIndex.from_documents(
                all_documents,
                storage_context=storage_context,
                show_progress=True
            )
            logger.info("Vector index created successfully")
        
        return index, None
    
    except Exception as e:
        logger.error(f"Failed to load and index documents: {e}", exc_info=True)
        return None, f"エラー: {str(e)}"


def query_index(index, query_text, similarity_top_k=3):
    """インデックスへのクエリ実行"""
    logger.info(f"Querying index: {query_text[:50]}...")
    
    try:
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="compact"
        )
        
        response = query_engine.query(query_text)
        logger.info(f"Query completed: {len(response.source_nodes)} sources found")
        return response
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise QueryError(f"検索に失敗しました: {e}")
