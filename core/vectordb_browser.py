"""
VectorDB可視化モジュール - メタデータ編集機能付き
登録されているテキストと画像の関係を表示 + メタデータ管理
"""
import json
import streamlit as st
from pathlib import Path
import chromadb
from utils.logger import get_logger
from core.metadata_utils import (
    get_file_list,
    get_file_metadata,
    update_file_metadata,
    bulk_update_metadata,
    preview_bulk_update
)

logger = get_logger()


def get_all_documents_from_vectordb(chroma_client, collection_name="multimodal_rag"):
    """
    VectorDBから全ドキュメントを取得
    """
    try:
        collections = chroma_client.list_collections()
        logger.info(f"Available collections: {[c.name for c in collections]}")
        
        if not any(c.name == collection_name for c in collections):
            if collections:
                collection_name = collections[0].name
                logger.info(f"Using collection: {collection_name}")
            else:
                logger.warning("No collections found in VectorDB")
                return []
        
        collection = chroma_client.get_collection(collection_name)
        
        results = collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        
        logger.info(f"Retrieved {len(results['ids'])} documents from VectorDB (collection: {collection_name})")
        
        documents = []
        for i in range(len(results['ids'])):
            if results['embeddings'] is not None and len(results['embeddings']) > 0:
                embedding_size = len(results['embeddings'][i]) if results['embeddings'][i] is not None else 0
            else:
                embedding_size = 0
            
            doc = {
                "id": results['ids'][i],
                "text": results['documents'][i],
                "metadata": results['metadatas'][i],
                "embedding_size": embedding_size
            }
            documents.append(doc)
        
        return documents
    
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def group_documents_by_file(documents):
    """
    ドキュメントをファイルごとにグループ化
    """
    grouped = {}
    
    for doc in documents:
        file_name = doc['metadata'].get('file_name', 'Unknown')
        if file_name not in grouped:
            grouped[file_name] = []
        grouped[file_name].append(doc)
    
    for file_name in grouped:
        grouped[file_name].sort(key=lambda x: x['metadata'].get('page', 0))
    
    return grouped


def render_vectordb_browser(chroma_client, image_cache):
    """
    VectorDBブラウザーUI - 3タブ構成
    """
    st.header("🔍 VectorDB ブラウザー")
    st.caption("登録されているテキストと画像の関係を確認 + メタデータ管理")
    
    documents = get_all_documents_from_vectordb(chroma_client)
    
    if not documents:
        st.warning("⚠️ VectorDBにドキュメントが登録されていません")
        st.info("「📚 ドキュメント管理」タブでインデックスを作成してください")
        return
    
    # 統計情報
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総チャンク数", len(documents))
    
    with col2:
        total_images = sum(doc['metadata'].get('num_images', 0) for doc in documents)
        st.metric("総画像数", total_images)
    
    with col3:
        unique_files = len(set(doc['metadata'].get('file_name') for doc in documents))
        st.metric("ファイル数", unique_files)
    
    with col4:
        avg_text_length = sum(len(doc['text']) for doc in documents) / len(documents)
        st.metric("平均文字数", f"{avg_text_length:.0f}")
    
    st.markdown("---")
    
    # 3タブ構成
    tab1, tab2, tab3 = st.tabs([
        "📖 閲覧モード",
        "✏️ 個別編集",
        "⚡ 一括管理"
    ])
    
    with tab1:
        render_browse_mode(chroma_client, documents, image_cache)
    
    with tab2:
        render_individual_editor(chroma_client)
    
    with tab3:
        render_bulk_manager(chroma_client)


def render_browse_mode(chroma_client, documents, image_cache):
    """閲覧モード（既存機能）"""
    grouped_docs = group_documents_by_file(documents)
    
    view_mode = st.radio(
        "表示モード",
        options=["ファイル別", "全ページ一覧", "画像付きのみ"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if view_mode == "ファイル別":
        render_by_file(grouped_docs, image_cache)
    elif view_mode == "全ページ一覧":
        render_all_pages(documents, image_cache)
    elif view_mode == "画像付きのみ":
        render_with_images_only(documents, image_cache)


def render_individual_editor(chroma_client):
    """個別編集タブ"""
    st.subheader("✏️ ファイル別メタデータ編集")
    st.caption("各ファイル固有の情報を個別に設定できます")
    
    # ファイル一覧取得
    files = get_file_list(chroma_client)
    
    if not files:
        st.info("ファイルがありません")
        return
    
    st.info(f"📊 全{len(files)}ファイル")
    
    # ファイルごとに編集UI
    for file_name, info in sorted(files.items()):
        with st.expander(
            f"📄 {file_name} ({info['page_count']}ページ / {info['chunk_count']}チャンク)",
            expanded=False
        ):
            render_file_editor(chroma_client, file_name, info)


def render_file_editor(chroma_client, file_name, info):
    """個別ファイルの編集UI"""
    metadata = info["metadata"]
    
    st.write("**現在のメタデータ:**")
    
    # 2カラムレイアウト
    col1, col2 = st.columns(2)
    
    with col1:
        product_type = st.text_input(
            "製品種別",
            value=metadata.get("product_type", ""),
            key=f"product_{file_name}",
            help="例: エアコン、洗濯機、冷蔵庫"
        )
        
        model = st.text_input(
            "型番",
            value=metadata.get("model", ""),
            key=f"model_{file_name}",
            help="例: A型、B型"
        )
        
        model_number = st.text_input(
            "品番",
            value=metadata.get("model_number", ""),
            key=f"number_{file_name}",
            help="例: RAS-X40K"
        )
    
    with col2:
        manufacturer = st.text_input(
            "メーカー",
            value=metadata.get("manufacturer", ""),
            key=f"manu_{file_name}",
            help="例: 〇〇電機"
        )
        
        category = st.text_input(
            "カテゴリ",
            value=metadata.get("category", ""),
            key=f"cat_{file_name}",
            help="例: 冷暖房機器、家電"
        )
        
        tags = st.text_input(
            "タグ（カンマ区切り）",
            value=metadata.get("tags", ""),
            key=f"tags_{file_name}",
            help="例: 業務用, 2023年モデル"
        )
    
    # 備考欄
    notes = st.text_area(
        "備考・メモ",
        value=metadata.get("notes", ""),
        key=f"notes_{file_name}",
        height=100,
        help="自由記述欄"
    )
    
    # 保存ボタン
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        if st.button(
            f"💾 このファイルに適用（{info['chunk_count']}チャンク）",
            key=f"save_{file_name}",
            type="primary",
            use_container_width=True
        ):
            try:
                new_metadata = {
                    "product_type": product_type,
                    "model": model,
                    "model_number": model_number,
                    "manufacturer": manufacturer,
                    "category": category,
                    "tags": tags,
                    "notes": notes
                }
                
                # 空の値は含めない
                new_metadata = {k: v for k, v in new_metadata.items() if v}
                
                count = update_file_metadata(chroma_client, file_name, new_metadata)
                st.success(f"✅ {count}チャンクを更新しました")
                st.balloons()
                
                # 少し待ってからリロード
                import time
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 更新失敗: {e}")
                logger.error(f"Metadata update failed for {file_name}: {e}")
    
    with col_btn2:
        # 現在のメタデータ表示
        with st.popover("📋 詳細"):
            st.json(metadata)


def render_bulk_manager(chroma_client):
    """一括管理タブ"""
    st.subheader("⚡ 一括メタデータ管理")
    st.caption("複数ファイルに共通の情報を効率的に設定できます")
    
    # ファイル一覧取得
    files = get_file_list(chroma_client)
    
    if not files:
        st.info("ファイルがありません")
        return
    
    st.write("**ファイル選択:**")
    
    # 全選択チェックボックス
    select_all = st.checkbox("☑️ 全て選択", value=False, key="select_all_bulk")
    
    # ファイル選択
    selected_files = []
    for file_name, info in sorted(files.items()):
        default_checked = select_all
        if st.checkbox(
            f"📄 {file_name} ({info['page_count']}ページ / {info['chunk_count']}チャンク)",
            value=default_checked,
            key=f"bulk_select_{file_name}"
        ):
            selected_files.append(file_name)
    
    if not selected_files:
        st.info("👆 ファイルを選択してください")
        return
    
    st.markdown("---")
    
    # 選択ファイルの統計
    total_chunks = sum(files[f]["chunk_count"] for f in selected_files)
    st.info(f"📊 選択: {len(selected_files)}ファイル / {total_chunks}チャンク")
    
    st.write("**共通メタデータ:**")
    
    # 共通メタデータ入力
    col1, col2 = st.columns(2)
    
    with col1:
        common_product = st.text_input(
            "製品種別",
            key="bulk_product",
            help="全選択ファイルに適用"
        )
        
        common_manufacturer = st.text_input(
            "メーカー",
            key="bulk_manufacturer"
        )
        
        common_category = st.text_input(
            "カテゴリ",
            key="bulk_category"
        )
    
    with col2:
        common_tags = st.text_input(
            "タグ（カンマ区切り）",
            key="bulk_tags"
        )
        
        common_notes = st.text_area(
            "備考・メモ",
            key="bulk_notes",
            height=100
        )
    
    # 共通メタデータを構築
    common_metadata = {}
    if common_product:
        common_metadata["product_type"] = common_product
    if common_manufacturer:
        common_metadata["manufacturer"] = common_manufacturer
    if common_category:
        common_metadata["category"] = common_category
    if common_tags:
        common_metadata["tags"] = common_tags
    if common_notes:
        common_metadata["notes"] = common_notes
    
    # プレビュー
    if common_metadata:
        st.markdown("---")
        st.write("**📋 プレビュー（更新内容）:**")
        
        try:
            preview = preview_bulk_update(chroma_client, selected_files, common_metadata)
            
            for item in preview:
                if item["changes"]:
                    with st.expander(f"📄 {item['file_name']} ({item['chunk_count']}チャンク)"):
                        for field, change in item["changes"].items():
                            old_val = change["old"] if change["old"] else "(空)"
                            new_val = change["new"] if change["new"] else "(空)"
                            st.write(f"**{field}:** `{old_val}` → `{new_val}`")
        
        except Exception as e:
            st.error(f"プレビュー生成エラー: {e}")
        
        st.markdown("---")
        
        # 一括適用ボタン
        col_apply1, col_apply2 = st.columns([2, 1])
        
        with col_apply1:
            if st.button(
                f"💾 {len(selected_files)}ファイルに一括適用（{total_chunks}チャンク）",
                type="primary",
                use_container_width=True
            ):
                try:
                    with st.spinner("更新中..."):
                        result = bulk_update_metadata(chroma_client, selected_files, common_metadata)
                    
                    st.success(f"✅ 成功: {result['total_updated']}チャンクを更新しました")
                    
                    # 詳細表示
                    with st.expander("📊 更新詳細"):
                        for file_name, count in result["files"].items():
                            st.write(f"• {file_name}: {count}チャンク")
                    
                    st.balloons()
                    
                    import time
                    time.sleep(2)
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ 一括更新失敗: {e}")
                    logger.error(f"Bulk update failed: {e}")
        
        with col_apply2:
            if st.button("🔄 リセット", use_container_width=True):
                st.rerun()
    else:
        st.info("👆 共通メタデータを入力してください")


def render_by_file(grouped_docs, image_cache):
    """ファイル別表示"""
    st.subheader("📁 ファイル別表示")
    
    file_names = sorted(grouped_docs.keys())
    selected_file = st.selectbox("ファイルを選択", file_names)
    
    if selected_file:
        docs = grouped_docs[selected_file]
        
        # ページごとにグループ化
        pages = {}
        for doc in docs:
            page_num = doc['metadata'].get('page', 0)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(doc)
        
        total_chunks = len(docs)
        total_pages = len(pages)
        st.info(f"📄 {selected_file}: {total_pages}ページ（{total_chunks}チャンク）")
        
        # ページ選択
        page_numbers = sorted(pages.keys())
        selected_page = st.selectbox(
            "ページを選択", 
            page_numbers,
            format_func=lambda x: f"ページ {x} ({len(pages[x])}チャンク)"
        )
        
        page_docs = pages[selected_page]
        
        # チャンクが複数ある場合はチャンク選択
        if len(page_docs) > 1:
            st.caption(f"このページには{len(page_docs)}個のチャンクがあります")
            selected_chunk_idx = st.selectbox(
                "チャンクを選択",
                range(len(page_docs)),
                format_func=lambda x: f"チャンク {x+1}/{len(page_docs)} ({len(page_docs[x]['text'])}文字)"
            )
            doc = page_docs[selected_chunk_idx]
        else:
            doc = page_docs[0]
        
        render_document_detail(doc, image_cache)


def render_all_pages(documents, image_cache):
    """全ページ一覧表示"""
    st.subheader("📄 全ページ一覧")
    
    items_per_page = st.slider("1ページあたりの表示件数", 1, 20, 5)
    total_pages = (len(documents) - 1) // items_per_page + 1
    current_page = st.number_input("ページ", 1, total_pages, 1)
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(documents))
    
    st.caption(f"全{len(documents)}件中 {start_idx + 1}-{end_idx}件を表示")
    
    for idx in range(start_idx, end_idx):
        doc = documents[idx]
        with st.expander(
            f"📄 {doc['metadata'].get('file_name', 'Unknown')} - "
            f"ページ {doc['metadata'].get('page', '?')} "
            f"({'🖼️' if doc['metadata'].get('num_images', 0) > 0 else '📝'})",
            expanded=False
        ):
            render_document_detail(doc, image_cache)
        st.markdown("---")


def render_with_images_only(documents, image_cache):
    """画像付きドキュメントのみ表示"""
    st.subheader("🖼️ 画像付きドキュメント")
    
    docs_with_images = [doc for doc in documents if doc['metadata'].get('num_images', 0) > 0]
    
    if not docs_with_images:
        st.warning("画像付きのドキュメントがありません")
        return
    
    st.info(f"全{len(documents)}件中、{len(docs_with_images)}件が画像付き")
    
    items_per_page = st.slider("1ページあたりの表示件数", 1, 10, 3)
    total_pages = (len(docs_with_images) - 1) // items_per_page + 1
    current_page = st.number_input("ページ", 1, total_pages, 1)
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(docs_with_images))
    
    for idx in range(start_idx, end_idx):
        doc = docs_with_images[idx]
        with st.expander(
            f"📄 {doc['metadata'].get('file_name', 'Unknown')} - "
            f"ページ {doc['metadata'].get('page', '?')} "
            f"({doc['metadata'].get('num_images', 0)}枚の画像)",
            expanded=True
        ):
            render_document_detail(doc, image_cache)
        st.markdown("---")


def render_document_detail(doc, image_cache):
    """ドキュメント詳細表示"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 テキスト内容")
        
        metadata = doc['metadata']
        st.markdown(f"**ファイル名:** {metadata.get('file_name', 'Unknown')}")
        st.markdown(f"**ページ:** {metadata.get('page', '?')} / {metadata.get('total_pages', '?')}")
        st.markdown(f"**画像数:** {metadata.get('num_images', 0)}枚")
        st.markdown(f"**文字数:** {len(doc['text'])}文字")
        
        with st.container():
            st.text_area(
                "内容",
                doc['text'],
                height=200,
                key=f"text_{doc['id']}"
            )
    
    with col2:
        st.markdown("### 🔢 メタデータ")
        st.json(metadata)
    
    # 画像表示
    if metadata.get('num_images', 0) > 0:
        st.markdown("---")
        st.markdown("### 🖼️ 関連画像")
        
        try:
            image_ids_str = metadata.get('image_ids', '[]')
            if isinstance(image_ids_str, str):
                image_ids = json.loads(image_ids_str)
            else:
                image_ids = image_ids_str
            
            if image_ids:
                cols = st.columns(min(3, len(image_ids)))
                
                for idx, image_id in enumerate(image_ids):
                    cached_data = image_cache.get_image(image_id)
                    
                    if cached_data:
                        with cols[idx % 3]:
                            st.image(
                                cached_data["image"],
                                caption=f"画像 {idx + 1}: {image_id}",
                                use_container_width=True
                            )
                            
                            img_meta = cached_data["metadata"]
                            st.caption(f"ページ: {img_meta.get('page')}")
                            st.caption(f"タイプ: {img_meta.get('type')}")
                    else:
                        with cols[idx % 3]:
                            st.error(f"❌ 画像が見つかりません: {image_id}")
            else:
                st.info("画像IDが記録されていません")
        
        except json.JSONDecodeError as e:
            st.error(f"画像ID解析エラー: {e}")
        except Exception as e:
            st.error(f"画像表示エラー: {e}")
    
    # デバッグ情報
    with st.expander("🔧 デバッグ情報"):
        st.markdown(f"**Document ID:** `{doc['id']}`")
        st.markdown(f"**Embedding次元数:** {doc['embedding_size']}")
        
        if 'image_ids' in metadata:
            st.markdown("**Image IDs (Raw):**")
            st.code(metadata['image_ids'])


def export_vectordb_summary(documents):
    """
    VectorDB内容をエクスポート
    """
    st.subheader("📥 データエクスポート")
    
    summary = []
    for doc in documents:
        summary.append({
            "file_name": doc['metadata'].get('file_name'),
            "page": doc['metadata'].get('page'),
            "text_length": len(doc['text']),
            "num_images": doc['metadata'].get('num_images', 0),
            "text_preview": doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
        })
    
    import json
    json_str = json.dumps(summary, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="📥 JSON形式でダウンロード",
        data=json_str,
        file_name="vectordb_summary.json",
        mime="application/json"
    )
    
    import csv
    import io
    
    csv_buffer = io.StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=summary[0].keys())
    csv_writer.writeheader()
    csv_writer.writerows(summary)
    
    st.download_button(
        label="📥 CSV形式でダウンロード",
        data=csv_buffer.getvalue(),
        file_name="vectordb_summary.csv",
        mime="text/csv"
    )