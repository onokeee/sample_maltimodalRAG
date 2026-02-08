"""
カスタム例外クラス
"""


class MultimodalRAGError(Exception):
    """基底例外クラス"""
    pass


class PDFProcessingError(MultimodalRAGError):
    """PDF処理エラー"""
    pass


class ImageExtractionError(MultimodalRAGError):
    """画像抽出エラー"""
    pass


class IndexCreationError(MultimodalRAGError):
    """インデックス作成エラー"""
    pass


class QueryError(MultimodalRAGError):
    """クエリ実行エラー"""
    pass


class FileUploadError(MultimodalRAGError):
    """ファイルアップロードエラー"""
    pass


class APIKeyError(MultimodalRAGError):
    """APIキーエラー"""
    pass
