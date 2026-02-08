"""
メタデータ管理ユーティリティ
ファイル単位でのメタデータ取得・更新機能
"""
from typing import Dict, List, Optional
from datetime import datetime
from utils.logger import get_logger

logger = get_logger()


def get_file_list(chroma_client, collection_name="multimodal_rag"):
    """
    VectorDBに登録されているファイル一覧を取得
    
    Returns:
        dict: {
            "file_name": {
                "chunk_count": int,
                "page_count": int,
                "metadata": dict
            }
        }
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        results = collection.get(include=["metadatas"])
        
        files = {}
        for metadata in results.get("metadatas", []):
            file_name = metadata.get("file_name", "Unknown")
            
            if file_name not in files:
                files[file_name] = {
                    "chunk_count": 0,
                    "pages": set(),
                    "metadata": metadata.copy()
                }
            
            files[file_name]["chunk_count"] += 1
            
            # ページ番号を記録
            page = metadata.get("page")
            if page is not None:
                files[file_name]["pages"].add(page)
        
        # ページセットをカウントに変換
        for file_name in files:
            files[file_name]["page_count"] = len(files[file_name]["pages"])
            del files[file_name]["pages"]
        
        logger.info(f"Retrieved file list: {len(files)} files")
        return files
    
    except Exception as e:
        logger.error(f"Failed to get file list: {e}")
        return {}


def get_file_metadata(chroma_client, file_name, collection_name="multimodal_rag"):
    """
    指定ファイルの現在のメタデータを取得
    
    Args:
        file_name: ファイル名
    
    Returns:
        dict: メタデータ（最初のチャンクのメタデータを返す）
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        results = collection.get(
            where={"file_name": file_name},
            limit=1,
            include=["metadatas"]
        )
        
        if results and results.get("metadatas"):
            return results["metadatas"][0]
        else:
            return {}
    
    except Exception as e:
        logger.error(f"Failed to get metadata for {file_name}: {e}")
        return {}


def update_file_metadata(chroma_client, file_name, new_metadata, collection_name="multimodal_rag"):
    """
    指定ファイルの全チャンクのメタデータを更新
    
    Args:
        file_name: ファイル名
        new_metadata: 更新するメタデータ（dict）
    
    Returns:
        int: 更新したチャンク数
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        
        # そのファイルの全チャンクを取得
        results = collection.get(
            where={"file_name": file_name},
            include=["metadatas"]
        )
        
        if not results or not results.get("ids"):
            logger.warning(f"No chunks found for file: {file_name}")
            return 0
        
        # 各チャンクのメタデータを更新
        updated_count = 0
        for doc_id, old_metadata in zip(results["ids"], results["metadatas"]):
            # 既存メタデータに新しいメタデータをマージ
            updated_metadata = old_metadata.copy()
            updated_metadata.update(new_metadata)
            
            # 更新日時を追加
            updated_metadata["updated_at"] = datetime.now().isoformat()
            
            # ChromaDBに更新
            collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )
            updated_count += 1
        
        logger.info(f"Updated metadata for {file_name}: {updated_count} chunks")
        return updated_count
    
    except Exception as e:
        logger.error(f"Failed to update metadata for {file_name}: {e}")
        raise


def bulk_update_metadata(chroma_client, file_names, common_metadata, collection_name="multimodal_rag"):
    """
    複数ファイルに共通のメタデータを一括適用
    
    Args:
        file_names: ファイル名のリスト
        common_metadata: 共通メタデータ（dict）
    
    Returns:
        dict: {
            "total_updated": int,
            "files": {
                "file_name": chunk_count
            }
        }
    """
    try:
        result = {
            "total_updated": 0,
            "files": {}
        }
        
        for file_name in file_names:
            count = update_file_metadata(chroma_client, file_name, common_metadata, collection_name)
            result["files"][file_name] = count
            result["total_updated"] += count
        
        logger.info(f"Bulk update completed: {result['total_updated']} chunks across {len(file_names)} files")
        return result
    
    except Exception as e:
        logger.error(f"Bulk update failed: {e}")
        raise


def preview_bulk_update(chroma_client, file_names, common_metadata, collection_name="multimodal_rag"):
    """
    一括更新のプレビューを生成
    
    Args:
        file_names: ファイル名のリスト
        common_metadata: 共通メタデータ（dict）
    
    Returns:
        list: [{
            "file_name": str,
            "chunk_count": int,
            "changes": {
                "field_name": {"old": value, "new": value}
            }
        }]
    """
    try:
        preview = []
        
        for file_name in file_names:
            # 現在のメタデータを取得
            current_metadata = get_file_metadata(chroma_client, file_name, collection_name)
            
            # チャンク数を取得
            collection = chroma_client.get_collection(collection_name)
            results = collection.get(where={"file_name": file_name})
            chunk_count = len(results.get("ids", []))
            
            # 変更点を抽出
            changes = {}
            for key, new_value in common_metadata.items():
                old_value = current_metadata.get(key, "")
                if old_value != new_value:
                    changes[key] = {
                        "old": old_value,
                        "new": new_value
                    }
            
            preview.append({
                "file_name": file_name,
                "chunk_count": chunk_count,
                "changes": changes
            })
        
        return preview
    
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        return []


def delete_metadata_field(chroma_client, file_name, field_name, collection_name="multimodal_rag"):
    """
    指定ファイルの特定のメタデータフィールドを削除
    
    Args:
        file_name: ファイル名
        field_name: 削除するフィールド名
    
    Returns:
        int: 更新したチャンク数
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        
        # そのファイルの全チャンクを取得
        results = collection.get(
            where={"file_name": file_name},
            include=["metadatas"]
        )
        
        if not results or not results.get("ids"):
            return 0
        
        # 各チャンクのメタデータからフィールドを削除
        updated_count = 0
        for doc_id, metadata in zip(results["ids"], results["metadatas"]):
            if field_name in metadata:
                updated_metadata = metadata.copy()
                del updated_metadata[field_name]
                
                collection.update(
                    ids=[doc_id],
                    metadatas=[updated_metadata]
                )
                updated_count += 1
        
        logger.info(f"Deleted field '{field_name}' from {file_name}: {updated_count} chunks")
        return updated_count
    
    except Exception as e:
        logger.error(f"Failed to delete field '{field_name}' from {file_name}: {e}")
        raise


def get_all_metadata_fields(chroma_client, collection_name="multimodal_rag"):
    """
    VectorDB内の全メタデータフィールドを取得
    
    Returns:
        set: フィールド名のセット
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        results = collection.get(include=["metadatas"], limit=100)
        
        fields = set()
        for metadata in results.get("metadatas", []):
            fields.update(metadata.keys())
        
        logger.info(f"Found {len(fields)} metadata fields")
        return fields
    
    except Exception as e:
        logger.error(f"Failed to get metadata fields: {e}")
        return set()