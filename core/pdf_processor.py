"""
PDF処理モジュール - エラーハンドリング強化版
"""
import pypdf
import pdfplumber
from pathlib import Path
from utils.logger import get_logger
from utils.exceptions import PDFProcessingError

logger = get_logger()


def extract_text_from_pdf(pdf_path):
    """PDFからページ単位でテキストを抽出（エラーハンドリング強化）"""
    page_texts = {}
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise PDFProcessingError(f"PDFファイルが見つかりません: {pdf_path}")
    
    if pdf_path.stat().st_size == 0:
        logger.error(f"PDF file is empty: {pdf_path}")
        raise PDFProcessingError(f"PDFファイルが空です: {pdf_path}")
    
    logger.info(f"Starting PDF text extraction: {pdf_path.name}")
    
    # 方法1: pdfplumber を試す
    try:
        logger.debug("Trying pdfplumber...")
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        page_texts[page_num] = page_text
                        logger.debug(f"Extracted page {page_num} with pdfplumber ({len(page_text)} chars)")
                except Exception as e:
                    logger.warning(f"pdfplumber failed on page {page_num}: {e}")
                    continue
        
        if page_texts:
            logger.info(f"pdfplumber succeeded: {len(page_texts)} pages extracted")
            return page_texts
    except pdfplumber.exceptions.PDFSyntaxError as e:
        logger.warning(f"pdfplumber syntax error: {e}")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
    
    # 方法2: pypdf を試す
    try:
        logger.debug("Trying pypdf...")
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            if pdf_reader.is_encrypted:
                logger.error(f"PDF is encrypted: {pdf_path}")
                raise PDFProcessingError(f"暗号化されたPDFは読み込めません: {pdf_path}")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        page_texts[page_num] = page_text
                        logger.debug(f"Extracted page {page_num} with pypdf ({len(page_text)} chars)")
                except Exception as e:
                    logger.warning(f"pypdf failed on page {page_num}: {e}")
                    continue
        
        if page_texts:
            logger.info(f"pypdf succeeded: {len(page_texts)} pages extracted")
            return page_texts
    except pypdf.errors.PdfReadError as e:
        logger.error(f"pypdf read error: {e}")
        raise PDFProcessingError(f"PDFの読み込みに失敗しました（破損している可能性があります）: {e}")
    except Exception as e:
        logger.error(f"pypdf failed: {e}")
    
    # 両方失敗
    if not page_texts:
        logger.error(f"All PDF extraction methods failed: {pdf_path}")
        raise PDFProcessingError(
            f"PDFからテキストを抽出できませんでした: {pdf_path.name}\n"
            "画像のみのPDFの可能性があります。OCR機能を使用するか、テキスト付きPDFを使用してください。"
        )
    
    return page_texts


def validate_pdf_file(pdf_path):
    """PDFファイルの検証"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise PDFProcessingError(f"ファイルが存在しません: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise PDFProcessingError(f"PDFファイルではありません: {pdf_path}")
    
    if pdf_path.stat().st_size == 0:
        raise PDFProcessingError(f"ファイルが空です: {pdf_path}")
    
    # ファイルサイズ制限（例: 100MB）
    max_size_mb = 100
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise PDFProcessingError(
            f"ファイルサイズが大きすぎます: {file_size_mb:.1f}MB（上限: {max_size_mb}MB）"
        )
    
    logger.info(f"PDF validation passed: {pdf_path.name} ({file_size_mb:.1f}MB)")
    return True
