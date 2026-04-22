"""
BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN (Text Preprocessing)
==================================================
Làm sạch và chuẩn hóa text trước khi đưa vào model ML.
Hỗ trợ cả tiếng Anh và tiếng Việt.

Cài thư viện:
    pip install nltk spacy pandas tqdm
    python -m spacy download en_core_web_sm
    
    Tiếng Việt (tùy chọn):
    pip install underthesea
"""

import re
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data (chạy 1 lần)
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

# Thử import spaCy
try:
    import spacy
    nlp_en = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    HAS_SPACY = False
    print("Warning: spaCy chưa cài hoặc chưa download model. Dùng NLTK thay thế.")

# Thử import underthesea (tiếng Việt)
try:
    from underthesea import word_tokenize as vn_tokenize
    HAS_UNDERTHESEA = True
except:
    HAS_UNDERTHESEA = False

# ========================
# CẤU HÌNH
# ========================
BASE_DIR = Path("data")
TEXT_DIR = BASE_DIR / "raw_text"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================
# TỪ DỪNG (STOPWORDS)
# ========================

# Stopwords tiếng Anh từ NLTK
EN_STOPWORDS = set(stopwords.words('english'))

# Thêm stopwords đặc thù cho báo cáo ESG (những từ xuất hiện nhiều nhưng không có ý nghĩa)
ESG_EXTRA_STOPWORDS_EN = {
    'page', 'report', 'annual', 'company', 'year', 'please', 'also',
    'may', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'first', 'second', 'third', 'table',
    'figure', 'appendix', 'note', 'see', 'refer', 'following',
    'including', 'related', 'based', 'accordance', 'within', 'across',
    'ensure', 'continue', 'approach', 'provide', 'information', 'data'
}
EN_STOPWORDS.update(ESG_EXTRA_STOPWORDS_EN)

# Stopwords tiếng Việt (tự định nghĩa)
VN_STOPWORDS = {
    'và', 'của', 'trong', 'các', 'là', 'có', 'để', 'được', 'với',
    'cho', 'này', 'đó', 'từ', 'theo', 'về', 'tại', 'ra', 'khi',
    'đã', 'sẽ', 'bởi', 'qua', 'lên', 'xuống', 'thì', 'mà', 'hay',
    'hoặc', 'nếu', 'nhưng', 'vì', 'nên', 'do', 'tới', 'đến', 'như',
    'không', 'chưa', 'đang', 'rằng', 'cũng', 'vẫn', 'đây', 'đều',
    'những', 'một', 'hai', 'ba', 'bốn', 'năm', 'nhiều', 'một số',
    'công ty', 'báo cáo', 'năm nay', 'hoạt động', 'thực hiện'
}

# ========================
# HÀM PREPROCESSING
# ========================

class TextPreprocessor:
    """
    Class xử lý text cho cả tiếng Anh và tiếng Việt.
    
    Các bước:
    1. Lowercase
    2. Xóa số, URLs, email, ký tự đặc biệt
    3. Tokenization
    4. Xóa stopwords
    5. Lemmatization (tiếng Anh) / chuẩn hóa (tiếng Việt)
    6. Lọc token quá ngắn/quá dài
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    # -------- TIẾNG ANH --------
    
    def clean_text_en(self, text: str) -> str:
        """Làm sạch text tiếng Anh cơ bản"""
        # Chuyển về chữ thường
        text = text.lower()
        
        # Xóa URLs
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'www\.\S+', ' ', text)
        
        # Xóa email
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Xóa số (giữ lại từ có cả chữ và số như "co2", "pm2.5")
        text = re.sub(r'\b\d+\.?\d*\b', ' ', text)
        
        # Xóa dấu câu và ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Xóa ký tự đơn lẻ (trừ các chữ cái quan trọng)
        text = re.sub(r'\b[a-z]\b', ' ', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_en(self, text: str) -> List[str]:
        """Tokenize và chuẩn hóa từ tiếng Anh"""
        # Dùng spaCy nếu có (chính xác hơn)
        if HAS_SPACY:
            doc = nlp_en(text[:1_000_000])  # Giới hạn để tránh quá chậm
            tokens = [
                token.lemma_.lower()
                for token in doc
                if (not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                    and token.is_alpha
                    and len(token.text) > 2)
            ]
        else:
            # Fallback: NLTK
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(t.lower()) for t in tokens
                     if t.isalpha() and len(t) > 2]
        
        # Lọc stopwords
        tokens = [t for t in tokens if t not in EN_STOPWORDS]
        
        # Lọc token quá ngắn hoặc quá dài
        tokens = [t for t in tokens if 2 < len(t) < 25]
        
        return tokens
    
    def preprocess_en(self, text: str) -> dict:
        """Pipeline đầy đủ cho tiếng Anh"""
        cleaned = self.clean_text_en(text)
        tokens = self.tokenize_en(cleaned)
        return {
            "tokens": tokens,
            "processed_text": ' '.join(tokens),
            "token_count": len(tokens)
        }
    
    # -------- TIẾNG VIỆT --------
    
    def clean_text_vn(self, text: str) -> str:
        """Làm sạch text tiếng Việt"""
        # Xóa URLs, email
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Xóa số
        text = re.sub(r'\b\d+[.,]?\d*\b', ' ', text)
        
        # Xóa ký tự đặc biệt (giữ lại ký tự tiếng Việt)
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_vn(self, text: str) -> List[str]:
        """Tokenize tiếng Việt"""
        if HAS_UNDERTHESEA:
            # underthesea xử lý từ ghép tiếng Việt tốt hơn
            tokens = vn_tokenize(text, format="text").split()
        else:
            # Fallback: tách theo khoảng trắng
            tokens = text.split()
        
        # Chuyển về chữ thường
        tokens = [t.lower() for t in tokens]
        
        # Xóa stopwords tiếng Việt
        tokens = [t for t in tokens if t not in VN_STOPWORDS]
        
        # Lọc token quá ngắn
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def preprocess_vn(self, text: str) -> dict:
        """Pipeline đầy đủ cho tiếng Việt"""
        cleaned = self.clean_text_vn(text)
        tokens = self.tokenize_vn(cleaned)
        return {
            "tokens": tokens,
            "processed_text": ' '.join(tokens),
            "token_count": len(tokens)
        }
    
    # -------- HÀM CHÍNH --------
    
    def preprocess(self, text: str, language: str = "EN") -> dict:
        """
        Tiền xử lý văn bản.
        
        Args:
            text: Văn bản gốc
            language: "EN" hoặc "VN"
        
        Returns:
            dict với tokens, processed_text, token_count
        """
        if language == "VN":
            return self.preprocess_vn(text)
        else:
            return self.preprocess_en(text)


# ========================
# XỬ LÝ TOÀN BỘ FILE
# ========================

def load_metadata() -> Optional[pd.DataFrame]:
    """Đọc file metadata.csv nếu có"""
    meta_path = BASE_DIR / "metadata.csv"
    if meta_path.exists():
        return pd.read_csv(meta_path)
    return None


def process_all_texts():
    """
    Tiền xử lý toàn bộ file text trong data/raw_text/
    """
    preprocessor = TextPreprocessor()
    metadata = load_metadata()
    
    all_records = []
    
    # Xử lý từng ngôn ngữ
    for lang in ["EN", "VN"]:
        text_subdir = TEXT_DIR / lang
        if not text_subdir.exists():
            continue
        
        txt_files = list(text_subdir.glob("*.txt"))
        if not txt_files:
            logger.warning(f"Không có file .txt trong {text_subdir}")
            continue
        
        logger.info(f"\n=== Xử lý {len(txt_files)} file {lang} ===")
        
        for txt_path in tqdm(txt_files, desc=f"Preprocessing {lang}"):
            # Đọc text
            with open(txt_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Kiểm tra text đủ dài không
            if len(raw_text) < 500:
                logger.warning(f"  File quá ngắn: {txt_path.name}")
                continue
            
            # Tiền xử lý
            result = preprocessor.preprocess(raw_text, language=lang)
            
            if result["token_count"] < 50:
                logger.warning(f"  Quá ít token sau xử lý: {txt_path.name}")
                continue
            
            # Lấy thông tin từ metadata nếu có
            company_info = {}
            if metadata is not None:
                row = metadata[metadata['filename'].str.contains(txt_path.stem, na=False)]
                if not row.empty:
                    company_info = row.iloc[0].to_dict()
            
            # Lưu kết quả
            record = {
                "doc_id": txt_path.stem,
                "language": lang,
                "raw_length": len(raw_text),
                "token_count": result["token_count"],
                "processed_text": result["processed_text"],
                **company_info
            }
            all_records.append(record)
            
            # Lưu file processed
            out_path = PROCESSED_DIR / f"{txt_path.stem}_processed.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(result["processed_text"])
    
    if not all_records:
        logger.error("Không có dữ liệu nào được xử lý!")
        return None
    
    # Tạo DataFrame tổng hợp
    df = pd.DataFrame(all_records)
    
    # Lưu CSV chính
    df.to_csv(PROCESSED_DIR / "corpus.csv", index=False)
    
    # Thống kê
    logger.info(f"""
=== KẾT QUẢ PREPROCESSING ===
Tổng tài liệu    : {len(df)}
Tiếng Anh        : {len(df[df['language'] == 'EN'])}
Tiếng Việt       : {len(df[df['language'] == 'VN'])}
Trung bình tokens: {df['token_count'].mean():.0f}
Min tokens       : {df['token_count'].min()}
Max tokens       : {df['token_count'].max()}
    """)
    
    return df


# Thống kê nhanh về vocabulary
def analyze_vocabulary(df: pd.DataFrame):
    """Phân tích từ vựng của corpus"""
    from collections import Counter
    
    all_tokens = []
    for text in df['processed_text']:
        all_tokens.extend(text.split())
    
    vocab = Counter(all_tokens)
    
    logger.info(f"\n=== THỐNG KÊ TỪ VỰNG ===")
    logger.info(f"Tổng từ vựng (unique): {len(vocab):,}")
    logger.info(f"Top 20 từ phổ biến nhất:")
    for word, count in vocab.most_common(20):
        logger.info(f"  {word}: {count:,}")
    
    return vocab


# ========================
# CHẠY SCRIPT
# ========================
if __name__ == "__main__":
    print("=" * 60)
    print("BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN")
    print("=" * 60)
    
    df = process_all_texts()
    
    if df is not None:
        vocab = analyze_vocabulary(df)
        print(f"\nHoàn thành! Corpus lưu tại: {PROCESSED_DIR / 'corpus.csv'}")
        print(f"Tổng: {len(df)} tài liệu, {len(vocab):,} từ unique")
