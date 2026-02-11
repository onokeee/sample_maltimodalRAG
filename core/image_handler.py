"""
画像処理モジュール - メモリ効率改善版
"""
import io
import tempfile
import hashlib
from pathlib import Path
from PIL import Image
import fitz
import streamlit as st
from utils.logger import get_logger
from utils.exceptions import ImageExtractionError

logger = get_logger()


class ImageCache:
    """画像キャッシュマネージャー（メモリ効率改善）"""
    
    def __init__(self, cache_dir="./image_cache", max_memory_mb=500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        self.registry = {}
        logger.info(f"ImageCache initialized: {cache_dir}, max_memory={max_memory_mb}MB")
    
    def _get_cache_path(self, image_id):
        """キャッシュファイルパスを取得"""
        hash_id = hashlib.md5(image_id.encode()).hexdigest()
        return self.cache_dir / f"{hash_id}.png"
    
    def add_image(self, image_id, image, metadata):
        """画像をキャッシュに追加（ディスク保存）"""
        try:
            cache_path = self._get_cache_path(image_id)
            
            # ディスクに保存
            image.save(cache_path, format="PNG", optimize=True)
            file_size = cache_path.stat().st_size
            
            # レジストリに記録
            self.registry[image_id] = {
                "path": str(cache_path),
                "metadata": metadata,
                "size": file_size
            }
            
            self.current_memory += file_size
            logger.info(f"Image cached: {image_id} ({file_size / 1024:.1f}KB)")
            
            # メモリ制限チェック
            if self.current_memory > self.max_memory_bytes:
                self._cleanup_old_images()
            
            return True
        except Exception as e:
            logger.error(f"Failed to cache image {image_id}: {e}")
            raise ImageExtractionError(f"画像キャッシュ失敗: {e}")
    
    def get_image(self, image_id):
        """画像を取得（必要に応じてディスクから読み込み）"""
        if image_id not in self.registry:
            logger.warning(f"Image not found in cache: {image_id}")
            return None
        
        try:
            cache_path = Path(self.registry[image_id]["path"])
            if not cache_path.exists():
                logger.error(f"Cache file missing: {cache_path}")
                del self.registry[image_id]
                return None
            
            image = Image.open(cache_path)
            return {
                "image": image,
                "metadata": self.registry[image_id]["metadata"]
            }
        except Exception as e:
            logger.error(f"Failed to load cached image {image_id}: {e}")
            return None
    
    def _cleanup_old_images(self):
        """古い画像を削除してメモリを解放"""
        logger.info("Cleaning up old cached images...")
        # 簡易実装: 全削除（本番では LRU など実装）
        for image_id in list(self.registry.keys())[:len(self.registry) // 2]:
            self._remove_image(image_id)
    
    def _remove_image(self, image_id):
        """画像を削除"""
        if image_id in self.registry:
            try:
                cache_path = Path(self.registry[image_id]["path"])
                if cache_path.exists():
                    cache_path.unlink()
                self.current_memory -= self.registry[image_id]["size"]
                del self.registry[image_id]
                logger.info(f"Image removed from cache: {image_id}")
            except Exception as e:
                logger.error(f"Failed to remove image {image_id}: {e}")
    
    def clear(self):
        """全キャッシュをクリア"""
        logger.info("Clearing all image cache...")
        for image_id in list(self.registry.keys()):
            self._remove_image(image_id)
        self.current_memory = 0


def extract_images_high_quality(pdf_path, dpi=300):
    """ページ全体を高品質画像化"""
    images = []
    logger.info(f"Extracting high-quality images from {pdf_path.name} at {dpi} DPI")
    
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        for page_num in range(total_pages):
            try:
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
                logger.debug(f"Extracted page {page_num + 1}/{total_pages}")
            except Exception as e:
                logger.error(f"Failed to extract page {page_num + 1}: {e}")
                continue
        
        pdf_document.close()
        logger.info(f"Successfully extracted {len(images)} page images")
    except Exception as e:
        logger.error(f"PDF document open failed: {e}")
        raise ImageExtractionError(f"PDF画像抽出エラー: {e}")
    
    return images


def extract_images_embedded_positioned(pdf_path, min_size=100):
    """位置情報ベースで画像を正確に切り抜く"""
    images = []
    logger.info(f"Extracting embedded images from {pdf_path.name} (min_size={min_size})")
    
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        for page_num in range(total_pages):
            try:
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
                            
                            # サイズフィルタ
                            if width < min_size or height < min_size:
                                continue
                            
                            # アスペクト比フィルタ
                            aspect_ratio = width / height if height > 0 else 0
                            if aspect_ratio > 10 or aspect_ratio < 0.1:
                                continue
                            
                            # 画像切り抜き
                            image = None
                            try:
                                mat = fitz.Matrix(2.0, 2.0)
                                clip_rect = fitz.Rect(x0, y0, x1, y1)
                                pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
                                img_data = pix.tobytes("png")
                                image = Image.open(io.BytesIO(img_data))
                            except Exception:
                                # フォールバック: 生データから取得（JBIG2/CMYK等の特殊形式対応）
                                try:
                                    base_image = pdf_document.extract_image(xref)
                                    if base_image and base_image.get("image"):
                                        image = Image.open(io.BytesIO(base_image["image"]))
                                        # CMYK等はRGBに変換
                                        if image.mode not in ("RGB", "RGBA", "L"):
                                            image = image.convert("RGB")
                                        logger.debug(f"Fallback extraction succeeded for xref={xref} on page {page_num + 1}")
                                except Exception as e2:
                                    logger.warning(f"Failed to extract image on page {page_num + 1} (xref={xref}): {e2}")
                                    continue
                            
                            if image is None:
                                continue
                            
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
                    except Exception as e:
                        logger.warning(f"Failed to extract embedded image on page {page_num + 1}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1}: {e}")
                continue
        
        pdf_document.close()
        logger.info(f"Successfully extracted {len(images)} embedded images")
    except Exception as e:
        logger.error(f"Embedded image extraction failed: {e}")
        raise ImageExtractionError(f"埋め込み画像抽出エラー: {e}")
    
    return images


def extract_images_from_pdf(pdf_path, method="high_quality", dpi=300):
    """画像抽出の統合関数"""
    logger.info(f"Starting image extraction: method={method}, dpi={dpi}")
    
    try:
        if method == "high_quality":
            return extract_images_high_quality(pdf_path, dpi=300)
        elif method == "medium_quality":
            return extract_images_high_quality(pdf_path, dpi=150)
        elif method == "embedded":
            return extract_images_embedded_positioned(pdf_path)
        elif method == "combined":
            page_images = extract_images_high_quality(pdf_path, dpi=200)
            embedded_images = extract_images_embedded_positioned(pdf_path)
            logger.info(f"Combined extraction: {len(page_images)} pages + {len(embedded_images)} embedded")
            return page_images + embedded_images
        else:
            logger.warning(f"Unknown method {method}, using high_quality")
            return extract_images_high_quality(pdf_path, dpi=300)
    except Exception as e:
        logger.error(f"Image extraction failed: {e}")
        raise ImageExtractionError(f"画像抽出に失敗しました: {e}")
