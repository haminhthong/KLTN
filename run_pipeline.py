"""
BƯỚC 7: CHẠY TOÀN BỘ PIPELINE
================================
Script tổng hợp để chạy từng bước hoặc tất cả.

Cách dùng:
    python run_pipeline.py --all          # Chạy tất cả
    python run_pipeline.py --step 1       # Chỉ chạy bước 1
    python run_pipeline.py --from-step 3  # Chạy từ bước 3 trở đi
"""

import argparse
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_step(step_num: int):
    """Chạy một bước cụ thể"""
    
    step_map = {
        1: ("CONVERT PDF → TEXT", "01_pdf_to_text", "convert_all_pdfs"),
        2: ("TIỀN XỬ LÝ VĂN BẢN", "02_preprocessing", "process_all_texts"),
        3: ("TF-IDF & KEYWORDS", "03_tfidf_keywords", "run_tfidf_pipeline"),
        4: ("LDA TOPIC MODELING", "04_lda_topic_modeling", "run_lda_pipeline"),
        5: ("SENTIMENT ANALYSIS", "05_sentiment_analysis", "run_sentiment_pipeline"),
        6: ("RANDOM FOREST + RIDGE", "06_modeling", "run_modeling_pipeline"),
    }
    
    if step_num not in step_map:
        logger.error(f"Bước {step_num} không tồn tại")
        return False
    
    step_name, module_name, func_name = step_map[step_num]
    
    print(f"\n{'='*60}")
    print(f"BƯỚC {step_num}: {step_name}")
    print(f"{'='*60}")
    
    start = time.time()
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        func()
        
        elapsed = time.time() - start
        logger.info(f"✓ Bước {step_num} hoàn thành ({elapsed:.1f}s)")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Kiểm tra lại tên file và thư viện đã cài chưa")
        return False
    except Exception as e:
        logger.error(f"Lỗi ở bước {step_num}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_requirements():
    """Kiểm tra thư viện cần thiết"""
    required = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "gensim": "gensim",
        "nltk": "nltk",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "scipy": "scipy",
        "joblib": "joblib",
    }
    
    optional = {
        "pdfplumber": "pdfplumber",
        "spacy": "spacy",
        "textblob": "textblob",
        "vaderSentiment": "vaderSentiment",
        "wordcloud": "wordcloud",
        "pyLDAvis": "pyLDAvis",
        "underthesea": "underthesea",
        "transformers": "transformers",
    }
    
    print("\n=== KIỂM TRA THƯ VIỆN ===\n")
    
    all_ok = True
    print("Bắt buộc:")
    for pkg, pip_name in required.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  ✗ {pip_name}  ← Cần cài: pip install {pip_name}")
            all_ok = False
    
    print("\nTùy chọn (cài thêm để tốt hơn):")
    for pkg, pip_name in optional.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  - {pip_name}  (pip install {pip_name})")
    
    if not all_ok:
        print("\n⚠️  Cần cài các thư viện bắt buộc trước!")
        print("   Chạy: pip install -r requirements.txt")
    else:
        print("\n✓ Tất cả thư viện bắt buộc đã sẵn sàng!")
    
    return all_ok


def check_data():
    """Kiểm tra cấu trúc dữ liệu"""
    print("\n=== KIỂM TRA DỮ LIỆU ===\n")
    
    en_dir = Path("data/raw_pdf/EN")
    vn_dir = Path("data/raw_pdf/VN")
    
    en_count = len(list(en_dir.glob("*.pdf"))) if en_dir.exists() else 0
    vn_count = len(list(vn_dir.glob("*.pdf"))) if vn_dir.exists() else 0
    
    print(f"Báo cáo tiếng Anh (data/raw_pdf/EN): {en_count} file")
    print(f"Báo cáo tiếng Việt (data/raw_pdf/VN): {vn_count} file")
    print(f"Tổng cộng: {en_count + vn_count} file")
    
    if en_count + vn_count == 0:
        print("\n⚠️  Chưa có dữ liệu PDF!")
        print("   Xem hướng dẫn: 00_dataset_guide.md")
    elif en_count + vn_count < 30:
        print(f"\n⚠️  Dữ liệu ít ({en_count+vn_count} file). Model sẽ không chính xác.")
        print("   Cố gắng thu thập ít nhất 100-300 báo cáo.")
    else:
        print(f"\n✓ Dữ liệu OK!")
    
    # Kiểm tra metadata
    meta_path = Path("data/metadata.csv")
    if meta_path.exists():
        import pandas as pd
        df = pd.read_csv(meta_path)
        print(f"\nMetadata: {len(df)} dòng, cột: {list(df.columns)}")
        if 'esg_score' in df.columns:
            print("  ✓ Có ESG score → dùng cho supervised learning")
        else:
            print("  ⚠️  Không có esg_score → sẽ tạo proxy labels")
    else:
        print("\n⚠️  Chưa có metadata.csv → sẽ tạo proxy labels")


def create_folder_structure():
    """Tạo cấu trúc thư mục"""
    dirs = [
        "data/raw_pdf/EN",
        "data/raw_pdf/VN",
        "data/raw_text/EN",
        "data/raw_text/VN",
        "data/processed",
        "data/features",
        "models/lda",
        "models/ml",
        "plots",
        "results"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Đã tạo cấu trúc thư mục")


def print_pipeline_overview():
    """In tổng quan pipeline"""
    print("""
╔══════════════════════════════════════════════════════╗
║         ESG TEXT MINING PIPELINE                     ║
║         Hỗ trợ: Tiếng Anh + Tiếng Việt              ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  BƯỚC 1: PDF → Text (pdfplumber + OCR)               ║
║     ↓                                                ║
║  BƯỚC 2: Text Preprocessing (NLTK/spaCy)             ║
║     ↓                                                ║
║  BƯỚC 3: TF-IDF + Keyword Extraction                 ║
║     ↓                                                ║
║  BƯỚC 4: LDA Topic Modeling (Gensim)                 ║
║     ↓                                                ║
║  BƯỚC 5: Sentiment Analysis (VADER/FinBERT)          ║
║     ↓                                                ║
║  BƯỚC 6: Random Forest + Ridge Regression            ║
║     ↓                                                ║
║  OUTPUT: Báo cáo kết quả + Biểu đồ                   ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ pipeline")
    parser.add_argument("--step", type=int, help="Chạy bước cụ thể (1-6)")
    parser.add_argument("--from-step", type=int, help="Chạy từ bước N đến cuối")
    parser.add_argument("--check", action="store_true", help="Kiểm tra thư viện và dữ liệu")
    parser.add_argument("--setup", action="store_true", help="Tạo cấu trúc thư mục")
    
    args = parser.parse_args()
    
    print_pipeline_overview()
    
    if args.setup:
        create_folder_structure()
    
    elif args.check:
        check_requirements()
        check_data()
    
    elif args.step:
        run_step(args.step)
    
    elif args.from_step:
        for step in range(args.from_step, 7):
            success = run_step(step)
            if not success:
                logger.error(f"Pipeline dừng tại bước {step}")
                break
    
    elif args.all:
        check_requirements()
        check_data()
        
        print("\nBắt đầu chạy toàn bộ pipeline...")
        total_start = time.time()
        
        for step in range(1, 7):
            success = run_step(step)
            if not success:
                logger.error(f"Pipeline dừng tại bước {step}")
                sys.exit(1)
        
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"✓ HOÀN THÀNH TOÀN BỘ PIPELINE!")
        print(f"  Thời gian: {total_time/60:.1f} phút")
        print(f"  Kết quả: results/model_results.json")
        print(f"  Biểu đồ: plots/")
        print(f"{'='*60}")
    
    else:
        parser.print_help()
        print("\n💡 Bắt đầu nhanh:")
        print("   python run_pipeline.py --setup    # Tạo thư mục")
        print("   python run_pipeline.py --check    # Kiểm tra")
        print("   python run_pipeline.py --all      # Chạy tất cả")
        print("   python run_pipeline.py --step 3   # Chỉ chạy bước 3")
