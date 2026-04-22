"""
BƯỚC 3: TF-IDF & TRÍCH XUẤT TỪ KHÓA
======================================
TF-IDF (Term Frequency – Inverse Document Frequency)
Đây là phương pháp tìm từ QUAN TRỌNG nhất trong mỗi tài liệu.

LÝ THUYẾT NGẮN GỌN:
  TF  = Tần suất từ trong 1 tài liệu / Tổng số từ trong tài liệu đó
  IDF = log(Tổng số tài liệu / Số tài liệu chứa từ này)
  TF-IDF = TF × IDF

  → Từ có TF-IDF cao = xuất hiện nhiều trong tài liệu NÀY
    nhưng ít trong các tài liệu KHÁC
  → Ví dụ: "carbon emission" cao trong báo cáo môi trường
    nhưng "company" cao ở mọi báo cáo → IDF thấp → TF-IDF thấp

Cài thư viện:
    pip install scikit-learn pandas numpy matplotlib seaborn wordcloud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import logging
from pathlib import Path
from scipy.sparse import save_npz
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# WordCloud (tùy chọn)
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except:
    HAS_WORDCLOUD = False
    print("Tip: pip install wordcloud để tạo word cloud đẹp hơn")

# ========================
# CẤU HÌNH
# ========================
BASE_DIR    = Path("data")
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR = BASE_DIR / "features"
PLOTS_DIR    = Path("plots")

for d in [FEATURES_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================
# LOAD DỮ LIỆU
# ========================

def load_corpus() -> pd.DataFrame:
    corpus_path = PROCESSED_DIR / "corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {corpus_path}. Chạy bước 2 trước!")
    df = pd.read_csv(corpus_path)
    # Đảm bảo không có NaN trong processed_text
    df = df.dropna(subset=['processed_text'])
    df = df[df['processed_text'].str.len() > 10]
    logger.info(f"Đã load {len(df)} tài liệu")
    return df


# ========================
# XÂY DỰNG TF-IDF MATRIX
# ========================

def build_tfidf_matrix(
    documents: list,
    language: str = "EN",
    max_features: int = 5000,
    min_df: int = 3,
    max_df: float = 0.85,
    ngram_range: tuple = (1, 2)
):
    """
    Xây dựng TF-IDF matrix từ corpus.
    
    Args:
        documents: List các văn bản đã tiền xử lý
        language: "EN" hoặc "VN"
        max_features: Giữ tối đa bao nhiêu từ (đặc trưng)
        min_df: Từ phải xuất hiện trong ít nhất bao nhiêu tài liệu
        max_df: Bỏ từ xuất hiện trong quá nhiều tài liệu (VD: 85%)
        ngram_range: (1,1)=unigram, (1,2)=unigram+bigram
    
    Returns:
        vectorizer, tfidf_matrix
    """
    logger.info(f"Đang xây dựng TF-IDF matrix cho {len(documents)} tài liệu...")
    logger.info(f"  max_features={max_features}, min_df={min_df}, max_df={max_df}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=True,      # Dùng log(1+TF) thay vì TF thô → giảm ảnh hưởng từ quá phổ biến
        strip_accents='unicode',
        token_pattern=r'\b[a-zA-ZÀ-ỹ][a-zA-ZÀ-ỹ]+\b'  # Chỉ lấy từ có ít nhất 2 ký tự
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    logger.info(f"  Ma trận TF-IDF: {tfidf_matrix.shape}")
    logger.info(f"  ({tfidf_matrix.shape[0]} tài liệu × {tfidf_matrix.shape[1]} đặc trưng)")
    
    return vectorizer, tfidf_matrix, feature_names


# ========================
# TRÍCH XUẤT TỪ KHÓA
# ========================

def extract_top_keywords_per_doc(
    tfidf_matrix,
    feature_names,
    doc_ids: list,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Tìm top N từ khóa quan trọng nhất cho TỪNG TÀI LIỆU.
    """
    all_keywords = []
    
    for i, doc_id in enumerate(doc_ids):
        # Lấy hàng i của ma trận (TF-IDF scores của tài liệu i)
        row = tfidf_matrix[i]
        
        # Tìm indices của top N giá trị lớn nhất
        if hasattr(row, 'toarray'):
            scores = row.toarray().flatten()
        else:
            scores = row.flatten()
        
        top_indices = scores.argsort()[-top_n:][::-1]
        
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                all_keywords.append({
                    "doc_id": doc_id,
                    "keyword": feature_names[idx],
                    "tfidf_score": round(scores[idx], 4),
                    "rank": rank + 1
                })
    
    return pd.DataFrame(all_keywords)


def extract_global_keywords(
    tfidf_matrix,
    feature_names,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Tìm top N từ khóa QUAN TRỌNG NHẤT trong toàn bộ corpus.
    Dùng trung bình TF-IDF của tất cả tài liệu.
    """
    # Trung bình TF-IDF theo cột
    avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    
    top_indices = avg_scores.argsort()[-top_n:][::-1]
    
    df_global = pd.DataFrame({
        "keyword": [feature_names[i] for i in top_indices],
        "avg_tfidf": [round(avg_scores[i], 4) for i in top_indices]
    })
    
    return df_global


def extract_esg_keywords_by_pillar(feature_names, tfidf_matrix) -> dict:
    """
    Nhóm từ khóa theo 3 trụ cột ESG: E, S, G.
    
    Returns:
        dict với keys 'Environmental', 'Social', 'Governance'
    """
    # Từ khóa seed cho từng trụ cột
    esg_seeds = {
        "Environmental": [
            "carbon", "emission", "climate", "energy", "renewable", "waste",
            "water", "environment", "greenhouse", "biodiversity", "pollution",
            "sustainability", "green", "solar", "wind", "recycl", "fossil",
            "deforestation", "co2", "methane", "ecosystem", "conservation",
            # Tiếng Việt
            "môi trường", "khí thải", "năng lượng", "tái tạo", "rác thải"
        ],
        "Social": [
            "employee", "worker", "community", "health", "safety", "diversity",
            "inclusion", "human rights", "labor", "training", "education",
            "welfare", "stakeholder", "supply chain", "gender", "wage",
            "accident", "injury", "turnover", "volunteer", "charity",
            # Tiếng Việt
            "nhân viên", "cộng đồng", "sức khỏe", "an toàn", "đa dạng"
        ],
        "Governance": [
            "board", "governance", "ethics", "compliance", "transparency",
            "accountability", "audit", "risk", "policy", "regulation",
            "corruption", "bribery", "shareholder", "disclosure", "director",
            "executive", "remuneration", "whistleblower", "tax", "legal",
            # Tiếng Việt
            "hội đồng", "quản trị", "đạo đức", "tuân thủ", "minh bạch"
        ]
    }
    
    pillar_keywords = {}
    feature_set = set(feature_names)
    
    for pillar, seeds in esg_seeds.items():
        matched = []
        for feat in feature_names:
            for seed in seeds:
                if seed in feat.lower():
                    matched.append(feat)
                    break
        pillar_keywords[pillar] = matched
        logger.info(f"  {pillar}: {len(matched)} từ khóa liên quan")
    
    return pillar_keywords


# ========================
# TRỰC QUAN HÓA
# ========================

def plot_top_keywords_bar(df_global: pd.DataFrame, top_n: int = 30, title: str = ""):
    """Vẽ biểu đồ cột top keywords"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_plot = df_global.head(top_n).sort_values('avg_tfidf')
    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(df_plot)))
    
    bars = ax.barh(df_plot['keyword'], df_plot['avg_tfidf'], color=colors)
    ax.set_xlabel('Average TF-IDF Score', fontsize=12)
    ax.set_title(title or f'Top {top_n} Keywords in ESG Reports', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Thêm giá trị lên bar
    for bar, val in zip(bars, df_plot['avg_tfidf']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tfidf_top_keywords.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Đã lưu: plots/tfidf_top_keywords.png")


def plot_esg_pillars_keywords(pillar_keywords: dict, tfidf_matrix, feature_names):
    """Vẽ biểu đồ keywords theo từng trụ cột E, S, G"""
    feat_list = list(feature_names)
    avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    feat_score = dict(zip(feat_list, avg_scores))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    colors = {'Environmental': '#2ecc71', 'Social': '#3498db', 'Governance': '#9b59b6'}
    
    for ax, (pillar, keywords) in zip(axes, pillar_keywords.items()):
        # Lấy scores cho keywords của trụ cột này
        scores = [(kw, feat_score.get(kw, 0)) for kw in keywords]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:15]
        
        if not scores:
            ax.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
            ax.set_title(pillar)
            continue
        
        words, vals = zip(*scores)
        y_pos = range(len(words))
        
        ax.barh(y_pos, vals, color=colors.get(pillar, 'gray'), alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_title(f'🌿 {pillar}' if pillar == 'Environmental'
                     else f'👥 {pillar}' if pillar == 'Social'
                     else f'🏛️ {pillar}',
                     fontsize=13, fontweight='bold', color=colors[pillar])
        ax.set_xlabel('Avg TF-IDF')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Top Keywords by ESG Pillar', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tfidf_esg_pillars.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Đã lưu: plots/tfidf_esg_pillars.png")


def plot_wordcloud(df_global: pd.DataFrame, title: str = "ESG Keywords"):
    """Vẽ Word Cloud"""
    if not HAS_WORDCLOUD:
        logger.warning("wordcloud chưa cài, bỏ qua bước này")
        return
    
    word_freq = dict(zip(df_global['keyword'], df_global['avg_tfidf']))
    
    wc = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='RdYlGn',
        max_words=100,
        prefer_horizontal=0.7,
        min_font_size=10
    ).generate_from_frequencies(word_freq)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wordcloud_esg.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Đã lưu: plots/wordcloud_esg.png")


# ========================
# PIPELINE CHÍNH
# ========================

def run_tfidf_pipeline():
    # 1. Load corpus
    df = load_corpus()
    documents = df['processed_text'].tolist()
    doc_ids   = df['doc_id'].tolist()
    
    # 2. Xây dựng TF-IDF (cho toàn bộ corpus)
    vectorizer, tfidf_matrix, feature_names = build_tfidf_matrix(
        documents,
        max_features=5000,
        min_df=max(2, len(documents) // 50),  # Tự động điều chỉnh theo corpus size
        max_df=0.85,
        ngram_range=(1, 2)
    )
    
    # 3. Lưu matrix (dùng ở bước sau)
    save_npz(FEATURES_DIR / "tfidf_matrix.npz", tfidf_matrix)
    np.save(FEATURES_DIR / "tfidf_feature_names.npy", feature_names)
    df[['doc_id', 'language']].to_csv(FEATURES_DIR / "doc_index.csv", index=False)
    
    # 4. Trích xuất từ khóa global
    df_global = extract_global_keywords(tfidf_matrix, feature_names, top_n=100)
    df_global.to_csv(FEATURES_DIR / "global_keywords.csv", index=False)
    
    logger.info("\nTop 20 từ khóa ESG phổ biến nhất:")
    print(df_global.head(20).to_string(index=False))
    
    # 5. Trích xuất từ khóa theo tài liệu
    df_doc_keywords = extract_top_keywords_per_doc(
        tfidf_matrix, feature_names, doc_ids, top_n=20
    )
    df_doc_keywords.to_csv(FEATURES_DIR / "doc_keywords.csv", index=False)
    
    # 6. Phân nhóm theo E, S, G
    pillar_keywords = extract_esg_keywords_by_pillar(feature_names, tfidf_matrix)
    
    # 7. Trực quan hóa
    logger.info("\nĐang vẽ biểu đồ...")
    plot_top_keywords_bar(df_global, top_n=30)
    plot_esg_pillars_keywords(pillar_keywords, tfidf_matrix, feature_names)
    plot_wordcloud(df_global)
    
    logger.info(f"""
=== KẾT QUẢ TF-IDF ===
Ma trận: {tfidf_matrix.shape[0]} tài liệu × {tfidf_matrix.shape[1]} đặc trưng
Từ khóa global: {len(df_global)}
Keywords/tài liệu: Top 20
Files đã lưu:
  - data/features/tfidf_matrix.npz
  - data/features/global_keywords.csv
  - data/features/doc_keywords.csv
  - plots/tfidf_top_keywords.png
  - plots/tfidf_esg_pillars.png
  - plots/wordcloud_esg.png
    """)
    
    return vectorizer, tfidf_matrix, feature_names, df_global


if __name__ == "__main__":
    print("=" * 60)
    print("BƯỚC 3: TF-IDF & TRÍCH XUẤT TỪ KHÓA")
    print("=" * 60)
    run_tfidf_pipeline()
