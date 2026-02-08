"""
ロガー設定モジュール
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


class AppLogger:
    """アプリケーション用ロガー"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger("multimodal_rag")
        self.logger.setLevel(logging.INFO)
        
        # ログディレクトリ作成
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        # ファイルハンドラー（日付付き）
        log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # フォーマット
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラー追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        """ロガーインスタンスを取得"""
        return self.logger


def get_logger():
    """グローバルロガーを取得"""
    return AppLogger().get_logger()
