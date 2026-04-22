"""
BƯỚC 1: CONVERT PDF → TEXT
===========================
Chuyển toàn bộ báo cáo ESG từ định dạng PDF sang text thuần.
Hỗ trợ cả file PDF có thể đọc (digital) và file scan (dùng OCR).

Cài thư viện:
    pip install pdfplumber pytesseract Pillow pdf2image pandas tqdm
    
    Nếu dùng OCR (file scan), cần cài thêm:
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Mac: brew install tesseract
    - Linux: sudo apt-get install tesseract-ocr
"""

import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Thử import các thư viện PDF
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber chưa cài. Chạy: pip install pdfplumber")

try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Warning: OCR chưa cài. Nếu có file scan, cài: pip install pytesseract pdf2image Pillow")

# ========================
# CẤU HÌNH ĐƯỜNG DẪN
# ========================
BASE_DIR = Path("data")
PDF_EN_DIR = BASE_DIR / "raw_pdf" / "EN"
PDF_VN_DIR = BASE_DIR / "raw_pdf" / "VN"
TEXT_DIR   = BASE_DIR / "raw_text"
LOG_FILE   = BASE_DIR / "conversion_log.json"

# Tạo thư mục nếu chưa có
for d in [PDF_EN_DIR, PDF_VN_DIR, TEXT_DIR, TEXT_DIR / "EN", TEXT_DIR / "VN"]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "conversion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========================
# HÀM CHUYỂN ĐỔI PDF
# ========================

def extract_text_pdfplumber(pdf_path: Path) -> str:
    """
    Trích xuất text từ PDF digital (có thể đọc được).
    Đây là phương pháp chính, nhanh và chính xác.
    """
    text_pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"  Đang đọc {len(pdf.pages)} trang...")
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    # Xóa header/footer thường lặp lại (dòng đầu và cuối mỗi trang)
                    lines = page_text.split('\n')
                    if len(lines) > 4:
                        lines = lines[1:-1]  # Bỏ dòng đầu và cuối (thường là header/footer)
                    text_pages.append('\n'.join(lines))
    except Exception as e:
        logger.error(f"  Lỗi pdfplumber: {e}")
        return ""
    
    full_text = '\n\n'.join(text_pages)
    return full_text


def extract_text_ocr(pdf_path: Path, lang: str = "eng") -> str:
    """
    Trích xuất text từ PDF scan bằng OCR (Tesseract).
    Dùng khi pdfplumber không đọc được (text quá ít).
    
    lang: "eng" cho tiếng Anh, "vie" cho tiếng Việt, "eng+vie" cho cả hai
    """
    if not HAS_OCR:
        logger.warning("  OCR chưa cài đặt, bỏ qua file này")
        return ""
    
    text_pages = []
    try:
        logger.info(f"  Đang dùng OCR (chậm hơn)...")
        images = convert_from_path(pdf_path, dpi=200)
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image, lang=lang)
            text_pages.append(page_text)
            if (i + 1) % 10 == 0:
                logger.info(f"    OCR xong {i+1}/{len(images)} trang")
    except Exception as e:
        logger.error(f"  Lỗi OCR: {e}")
        return ""
    
    return '\n\n'.join(text_pages)


def clean_raw_text(text: str) -> str:
    """
    Làm sạch text thô ngay sau khi convert.
    (Chưa phải preprocessing NLP, chỉ xóa ký tự rác)
    """
    # Xóa ký tự không in được
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F]', ' ', text)
    
    # Chuẩn hóa xuống dòng
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Xóa dòng chỉ có số (thường là số trang)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Xóa khoảng trắng thừa
    text = re.sub(r' {3,}', ' ', text)      # 3+ dấu cách → 1 dấu cách
    text = re.sub(r'\n{4,}', '\n\n\n', text) # 4+ dòng trống → 3 dòng trống
    
    return text.strip()


def is_scanned_pdf(text: str, min_chars_per_page: int = 100) -> bool:
    """
    Kiểm tra xem PDF có phải file scan không.
    Nếu text quá ít → có thể là scan → cần OCR.
    """
    return len(text) < min_chars_per_page


def convert_pdf_to_text(
    pdf_path: Path,
    output_dir: Path,
    language: str = "EN",
    force_ocr: bool = False
) -> dict:
    """
    Hàm chính: Convert 1 file PDF → text.
    
    Args:
        pdf_path: Đường dẫn file PDF
        output_dir: Thư mục lưu file text
        language: "EN" hoặc "VN"
        force_ocr: True = luôn dùng OCR dù có text
    
    Returns:
        dict: Kết quả conversion
    """
    result = {
        "filename": pdf_path.name,
        "status": "pending",
        "method": None,
        "char_count": 0,
        "page_count": 0,
        "error": None
    }
    
    # Tên file output
    output_path = output_dir / pdf_path.stem  # VD: Unilever_2022_EN
    output_path = output_path.with_suffix('.txt')
    
    # Bỏ qua nếu đã convert
    if output_path.exists():
        logger.info(f"  Đã tồn tại: {output_path.name}, bỏ qua")
        result["status"] = "skipped"
        return result
    
    logger.info(f"Đang xử lý: {pdf_path.name}")
    
    # Bước 1: Thử pdfplumber trước
    text = ""
    if HAS_PDFPLUMBER and not force_ocr:
        text = extract_text_pdfplumber(pdf_path)
        result["method"] = "pdfplumber"
    
    # Bước 2: Nếu text quá ít → dùng OCR
    if is_scanned_pdf(text):
        logger.warning(f"  Text quá ít ({len(text)} ký tự), thử OCR...")
        ocr_lang = "vie+eng" if language == "VN" else "eng"
        text = extract_text_ocr(pdf_path, lang=ocr_lang)
        result["method"] = "ocr"
    
    # Kiểm tra kết quả
    if not text or len(text) < 50:
        logger.error(f"  Không đọc được text từ file này")
        result["status"] = "failed"
        result["error"] = "Cannot extract text"
        return result
    
    # Làm sạch text thô
    text = clean_raw_text(text)
    
    # Lưu file text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    result["status"] = "success"
    result["char_count"] = len(text)
    result["output_path"] = str(output_path)
    
    logger.info(f"  ✓ Xong: {len(text):,} ký tự → {output_path.name}")
    return result


def convert_all_pdfs():
    """
    Convert toàn bộ PDF trong thư mục data/raw_pdf/
    """
    all_results = []
    
    # Xử lý báo cáo tiếng Anh
    en_pdfs = list(PDF_EN_DIR.glob("*.pdf"))
    vn_pdfs = list(PDF_VN_DIR.glob("*.pdf"))
    
    logger.info(f"Tìm thấy {len(en_pdfs)} PDF tiếng Anh, {len(vn_pdfs)} PDF tiếng Việt")
    
    # Convert tiếng Anh
    if en_pdfs:
        logger.info("\n=== Xử lý báo cáo tiếng Anh ===")
        for pdf_path in tqdm(en_pdfs, desc="EN PDFs"):
            result = convert_pdf_to_text(
                pdf_path,
                output_dir=TEXT_DIR / "EN",
                language="EN"
            )
            result["language"] = "EN"
            all_results.append(result)
    
    # Convert tiếng Việt
    if vn_pdfs:
        logger.info("\n=== Xử lý báo cáo tiếng Việt ===")
        for pdf_path in tqdm(vn_pdfs, desc="VN PDFs"):
            result = convert_pdf_to_text(
                pdf_path,
                output_dir=TEXT_DIR / "VN",
                language="VN"
            )
            result["language"] = "VN"
            all_results.append(result)
    
    # Tổng kết
    df_results = pd.DataFrame(all_results)
    
    success = len(df_results[df_results['status'] == 'success'])
    failed = len(df_results[df_results['status'] == 'failed'])
    skipped = len(df_results[df_results['status'] == 'skipped'])
    
    logger.info(f"""
=== KẾT QUẢ CONVERT ===
✓ Thành công : {success}
✗ Thất bại   : {failed}
→ Đã bỏ qua  : {skipped}
Tổng          : {len(df_results)}
    """)
    
    # Lưu log
    df_results.to_csv(BASE_DIR / "conversion_results.csv", index=False)
    
    if failed > 0:
        logger.warning("Các file thất bại:")
        for _, row in df_results[df_results['status'] == 'failed'].iterrows():
            logger.warning(f"  - {row['filename']}: {row['error']}")
    
    return df_results


# ========================
# CHẠY SCRIPT
# ========================
if __name__ == "__main__":
    print("=" * 60)
    print("BƯỚC 1: CONVERT PDF → TEXT")
    print("=" * 60)
    print(f"Thư mục PDF tiếng Anh: {PDF_EN_DIR.absolute()}")
    print(f"Thư mục PDF tiếng Việt: {PDF_VN_DIR.absolute()}")
    print(f"Thư mục output: {TEXT_DIR.absolute()}")
    print()
    
    results = convert_all_pdfs()
    print("\nHoàn thành! Kiểm tra file: data/conversion_results.csv")
