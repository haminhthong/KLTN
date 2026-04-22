"""
BƯỚC 5: PHÂN TÍCH CẢM XÚC (Sentiment Analysis)
================================================
Phân tích tone/cảm xúc của từng báo cáo ESG:
  Positive → Công ty đang nói tốt về thành tích ESG
  Neutral  → Trung lập, mô tả khách quan
  Negative → Thừa nhận vấn đề, thách thức

LÝ THUYẾT:
  3 phương pháp chính:

  1. VADER (Valence Aware Dictionary and sEntiment Reasoner)
     - Dựa trên lexicon (từ điển cảm xúc)
     - Không cần training
     - Tốt cho văn bản ngắn, social media
     - Nhanh, đơn giản

  2. TextBlob
     - Tương tự VADER, dựa trên pattern
     - Cho ra Polarity (-1 đến 1) và Subjectivity (0 đến 1)

  3. FinBERT (khuyến nghị cho báo cáo tài chính)
     - BERT được fine-tune trên văn bản tài chính
     - Chính xác hơn nhiều cho ESG reports
     - Cần GPU hoặc RAM lớn

Cài thư viện:
    pip install vaderSentiment textblob transformers torch
    (transformers + torch chỉ cần nếu dùng FinBERT)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        HAS_VADER = True
    except:
        HAS_VADER = False
        print("Warning: VADER chưa cài. pip install vaderSentiment")

# TextBlob
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except:
    HAS_TEXTBLOB = False

# FinBERT (tùy chọn, cần nhiều RAM)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_FINBERT = True
except:
    HAS_FINBERT = False
    print("Tip: pip install transformers torch để dùng FinBERT (chính xác hơn)")

# ========================
# CẤU HÌNH
# ========================
BASE_DIR      = Path("data")
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR  = BASE_DIR / "features"
PLOTS_DIR     = Path("plots")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ESG-specific sentiment lexicon (bổ sung cho VADER)
ESG_POSITIVE_WORDS = {
    "improve", "increase", "achieve", "commit", "reduce", "sustainable",
    "progress", "target", "goal", "initiative", "success", "positive",
    "innovative", "responsible", "transparent", "efficient", "strong",
    "growth", "opportunity", "benefit", "enhance", "support", "lead",
    "award", "recognition", "milestone", "record", "best", "excellent"
}

ESG_NEGATIVE_WORDS = {
    "decrease", "fail", "challenge", "risk", "concern", "issue",
    "incident", "accident", "violation", "penalty", "fine", "shortage",
    "problem", "deficiency", "poor", "low", "insufficient", "delay",
    "loss", "decline", "deteriorate", "harm", "damage", "breach"
}


# ========================
# VADER SENTIMENT
# ========================

class VADERAnalyzer:
    """Phân tích sentiment bằng VADER"""
    
    def __init__(self):
        if not HAS_VADER:
            raise ImportError("VADER chưa cài đặt")
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Bổ sung ESG-specific words vào lexicon
        self.analyzer.lexicon.update({word: 2.0 for word in ESG_POSITIVE_WORDS})
        self.analyzer.lexicon.update({word: -2.0 for word in ESG_NEGATIVE_WORDS})
    
    def analyze_text(self, text: str) -> dict:
        """Phân tích 1 văn bản"""
        scores = self.analyzer.polarity_scores(text)
        
        # compound: từ -1 (negative) đến +1 (positive)
        compound = scores['compound']
        
        # Phân loại
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        
        return {
            "compound": round(compound, 4),
            "positive": round(scores['pos'], 4),
            "neutral": round(scores['neu'], 4),
            "negative": round(scores['neg'], 4),
            "sentiment_label": label,
            "method": "VADER"
        }
    
    def analyze_by_sentences(self, text: str, max_sentences: int = 200) -> dict:
        """
        Phân tích theo từng câu rồi tổng hợp.
        Chính xác hơn cho văn bản dài như ESG reports.
        """
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = text.split('.')
        
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        sentences = sentences[:max_sentences]  # Giới hạn số câu
        
        if not sentences:
            return self.analyze_text(text)
        
        sentence_results = [self.analyzer.polarity_scores(s) for s in sentences]
        
        avg_compound = np.mean([r['compound'] for r in sentence_results])
        avg_pos = np.mean([r['pos'] for r in sentence_results])
        avg_neu = np.mean([r['neu'] for r in sentence_results])
        avg_neg = np.mean([r['neg'] for r in sentence_results])
        
        if avg_compound >= 0.05:
            label = "Positive"
        elif avg_compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        
        return {
            "compound": round(avg_compound, 4),
            "positive": round(avg_pos, 4),
            "neutral": round(avg_neu, 4),
            "negative": round(avg_neg, 4),
            "sentiment_label": label,
            "n_sentences": len(sentences),
            "method": "VADER_sentence"
        }


# ========================
# TEXTBLOB SENTIMENT
# ========================

class TextBlobAnalyzer:
    """Phân tích sentiment bằng TextBlob"""
    
    def analyze_text(self, text: str) -> dict:
        if not HAS_TEXTBLOB:
            raise ImportError("TextBlob chưa cài đặt")
        
        # Giới hạn độ dài để tránh quá chậm
        text_sample = text[:50000]
        blob = TextBlob(text_sample)
        
        polarity = blob.sentiment.polarity        # -1 đến 1
        subjectivity = blob.sentiment.subjectivity  # 0 đến 1
        
        if polarity > 0.05:
            label = "Positive"
        elif polarity < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        
        return {
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
            "sentiment_label": label,
            "method": "TextBlob"
        }


# ========================
# FINBERT (OPTIONAL - BEST QUALITY)
# ========================

class FinBERTAnalyzer:
    """
    Phân tích sentiment bằng FinBERT.
    Được fine-tune trên dữ liệu tài chính → chính xác hơn.
    Cần RAM nhiều hơn (~2GB cho model).
    """
    
    def __init__(self):
        if not HAS_FINBERT:
            raise ImportError("transformers/torch chưa cài đặt")
        
        logger.info("Loading FinBERT model (lần đầu có thể mất vài phút download)...")
        self.pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",  # Model FinBERT public
            tokenizer="ProsusAI/finbert",
            device=-1  # CPU; dùng 0 nếu có GPU
        )
        logger.info("FinBERT loaded!")
    
    def analyze_text(self, text: str, chunk_size: int = 512) -> dict:
        """
        FinBERT chỉ xử lý được 512 tokens/lần.
        Chia text thành chunks và lấy trung bình.
        """
        # Chia text thành các đoạn nhỏ
        words = text.split()
        chunks = []
        for i in range(0, min(len(words), 5000), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        
        if not chunks:
            return {"sentiment_label": "Neutral", "score": 0.5, "method": "FinBERT"}
        
        # Phân tích từng chunk
        results = self.pipe(chunks[:10])  # Giới hạn 10 chunks
        
        # Tổng hợp
        label_scores = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            label = r['label'].lower()
            label_scores[label] += r['score']
        
        # Normalize
        total = sum(label_scores.values())
        if total > 0:
            label_scores = {k: v/total for k, v in label_scores.items()}
        
        dominant = max(label_scores, key=label_scores.get)
        
        return {
            "positive_score": round(label_scores['positive'], 4),
            "negative_score": round(label_scores['negative'], 4),
            "neutral_score": round(label_scores['neutral'], 4),
            "sentiment_label": dominant.capitalize(),
            "compound": round(label_scores['positive'] - label_scores['negative'], 4),
            "method": "FinBERT"
        }


# ========================
# PHÂN TÍCH CẢ CORPUS
# ========================

def analyze_corpus_sentiment(df: pd.DataFrame, method: str = "vader") -> pd.DataFrame:
    """
    Phân tích sentiment toàn bộ corpus.
    
    method: "vader", "textblob", "finbert", "all"
    """
    # Đọc raw text (chưa processed) để sentiment chính xác hơn
    # Sentiment cần text gốc, không phải text đã tokenize
    raw_text_dir = Path("data") / "raw_text"
    
    results = []
    
    if method in ("vader", "all") and HAS_VADER:
        vader = VADERAnalyzer()
    
    if method in ("textblob", "all") and HAS_TEXTBLOB:
        textblob_analyzer = TextBlobAnalyzer()
    
    if method in ("finbert", "all") and HAS_FINBERT:
        finbert = FinBERTAnalyzer()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment Analysis"):
        doc_id = row['doc_id']
        language = row.get('language', 'EN')
        
        # Đọc raw text (tốt hơn processed_text cho sentiment)
        raw_path = raw_text_dir / language / f"{doc_id}.txt"
        if raw_path.exists():
            with open(raw_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = row['processed_text']  # Fallback
        
        record = {"doc_id": doc_id, "language": language}
        
        # VADER
        if method in ("vader", "all") and HAS_VADER:
            if language == "EN":
                vader_result = vader.analyze_by_sentences(text)
            else:
                vader_result = vader.analyze_text(text)  # VADER kém tiếng Việt hơn
            record.update({
                "vader_compound": vader_result["compound"],
                "vader_pos": vader_result["positive"],
                "vader_neu": vader_result["neutral"],
                "vader_neg": vader_result["negative"],
                "vader_label": vader_result["sentiment_label"]
            })
        
        # TextBlob
        if method in ("textblob", "all") and HAS_TEXTBLOB and language == "EN":
            tb_result = textblob_analyzer.analyze_text(text)
            record.update({
                "tb_polarity": tb_result["polarity"],
                "tb_subjectivity": tb_result["subjectivity"],
                "tb_label": tb_result["sentiment_label"]
            })
        
        # FinBERT
        if method in ("finbert", "all") and HAS_FINBERT and language == "EN":
            fb_result = finbert.analyze_text(text)
            record.update({
                "finbert_compound": fb_result["compound"],
                "finbert_label": fb_result["sentiment_label"]
            })
        
        # Label cuối cùng (ensemble nếu có nhiều method)
        labels = []
        if "vader_label" in record:
            labels.append(record["vader_label"])
        if "finbert_label" in record:
            labels.append(record["finbert_label"])
        if labels:
            # Majority vote
            from collections import Counter
            record["final_sentiment"] = Counter(labels).most_common(1)[0][0]
        
        results.append(record)
    
    return pd.DataFrame(results)


# ========================
# TRỰC QUAN HÓA
# ========================

def plot_sentiment_overview(df_sentiment: pd.DataFrame):
    """Biểu đồ tổng quan phân bố sentiment"""
    
    label_col = "final_sentiment" if "final_sentiment" in df_sentiment.columns else "vader_label"
    if label_col not in df_sentiment.columns:
        logger.warning("Không có cột sentiment label")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
    
    # 1. Pie chart tổng
    counts = df_sentiment[label_col].value_counts()
    ax = axes[0]
    wedge_colors = [colors.get(l, 'gray') for l in counts.index]
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           colors=wedge_colors, startangle=90)
    ax.set_title("Phân bố Sentiment\n(Tổng corpus)", fontsize=12, fontweight='bold')
    
    # 2. Bar chart theo ngôn ngữ (nếu có)
    ax = axes[1]
    if 'language' in df_sentiment.columns:
        lang_sentiment = df_sentiment.groupby(['language', label_col]).size().unstack(fill_value=0)
        lang_sentiment.plot(kind='bar', ax=ax,
                           color=[colors.get(c, 'gray') for c in lang_sentiment.columns])
        ax.set_title("Sentiment theo Ngôn ngữ", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(title="Sentiment")
        ax.grid(axis='y', alpha=0.3)
    else:
        axes[1].set_visible(False)
    
    # 3. Distribution của compound score
    ax = axes[2]
    if "vader_compound" in df_sentiment.columns:
        ax.hist(df_sentiment["vader_compound"].dropna(), bins=30,
                color='steelblue', alpha=0.8, edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Neutral boundary')
        ax.axvline(df_sentiment["vader_compound"].mean(), color='green',
                  linestyle='-', linewidth=2, label=f"Mean = {df_sentiment['vader_compound'].mean():.3f}")
        ax.set_title("Phân phối VADER Compound Score", fontsize=12, fontweight='bold')
        ax.set_xlabel("Compound Score (-1 đến +1)")
        ax.set_ylabel("Số tài liệu")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sentiment_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Đã lưu: plots/sentiment_overview.png")


def plot_sentiment_by_company(df_sentiment: pd.DataFrame, df_meta: pd.DataFrame):
    """Top companies theo sentiment score"""
    if 'company' not in df_meta.columns:
        return
    
    df_merged = df_sentiment.merge(df_meta[['doc_id', 'company']], on='doc_id', how='left')
    
    if "vader_compound" not in df_merged.columns:
        return
    
    company_avg = df_merged.groupby('company')['vader_compound'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(company_avg) * 0.3)))
    colors = ['#e74c3c' if v < 0 else '#2ecc71' if v > 0.1 else '#95a5a6'
              for v in company_avg.values]
    
    ax.barh(company_avg.index, company_avg.values, color=colors, alpha=0.8)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel("Average VADER Compound Score")
    ax.set_title("Sentiment trung bình theo Công ty", fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sentiment_by_company.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Đã lưu: plots/sentiment_by_company.png")


# ========================
# PIPELINE CHÍNH
# ========================

def run_sentiment_pipeline(method: str = "vader"):
    """
    method: "vader" (nhanh), "textblob", "finbert" (chính xác, chậm), "all"
    """
    # 1. Load corpus
    df = pd.read_csv(PROCESSED_DIR / "corpus.csv").dropna(subset=['processed_text'])
    logger.info(f"Đã load {len(df)} tài liệu")
    
    # 2. Phân tích sentiment
    logger.info(f"\nPhương pháp: {method}")
    df_sentiment = analyze_corpus_sentiment(df, method=method)
    
    # 3. Thống kê
    label_col = "final_sentiment" if "final_sentiment" in df_sentiment.columns else "vader_label"
    if label_col in df_sentiment.columns:
        counts = df_sentiment[label_col].value_counts()
        logger.info(f"\nPhân bố Sentiment:")
        for label, count in counts.items():
            pct = count / len(df_sentiment) * 100
            logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    if "vader_compound" in df_sentiment.columns:
        logger.info(f"\nVADER Compound Score:")
        logger.info(f"  Mean: {df_sentiment['vader_compound'].mean():.4f}")
        logger.info(f"  Std:  {df_sentiment['vader_compound'].std():.4f}")
    
    # 4. Lưu kết quả
    df_sentiment.to_csv(FEATURES_DIR / "sentiment_scores.csv", index=False)
    
    # 5. Visualization
    plot_sentiment_overview(df_sentiment)
    
    # Nếu có metadata
    meta_path = BASE_DIR / "metadata.csv"
    if meta_path.exists():
        df_meta = pd.read_csv(meta_path)
        if 'doc_id' in df_meta.columns:
            plot_sentiment_by_company(df_sentiment, df_meta)
    
    logger.info(f"""
=== KẾT QUẢ SENTIMENT ===
Tổng tài liệu: {len(df_sentiment)}
Files đã lưu:
  - data/features/sentiment_scores.csv
  - plots/sentiment_overview.png
    """)
    
    return df_sentiment


if __name__ == "__main__":
    print("=" * 60)
    print("BƯỚC 5: SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Chọn method: "vader" (nhanh) hoặc "finbert" (chính xác)
    run_sentiment_pipeline(method="vader")
