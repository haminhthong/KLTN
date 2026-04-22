"""
BƯỚC 4: LDA TOPIC MODELING
============================
LDA (Latent Dirichlet Allocation) – Phân tích chủ đề ẩn

LÝ THUYẾT NGẮN GỌN:
  LDA giả định mỗi tài liệu = hỗn hợp của nhiều CHỦ ĐỀ
  Mỗi chủ đề = phân phối xác suất trên các TỪ

  Ví dụ với ESG:
  Tài liệu về năng lượng tái tạo:
    70% Topic "Environment" + 20% Topic "Business" + 10% Topic "Social"
  
  Model tự học ra:
    Topic 1 (Environment): carbon, emission, energy, climate, renewable...
    Topic 2 (Social): employee, safety, health, community, diversity...
    Topic 3 (Governance): board, ethics, compliance, audit, risk...

  Cách chọn số topic K:
    - Dùng Coherence Score: đo mức độ "có nghĩa" của topic
    - K tốt nhất = K có coherence score cao nhất

Cài thư viện:
    pip install gensim pyLDAvis matplotlib pandas numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from tqdm import tqdm

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

# pyLDAvis cho visualization đẹp (tùy chọn)
try:
    import pyLDAvis
    import pyLDAvis.gensim_models
    HAS_PYLDAVIS = True
except:
    HAS_PYLDAVIS = False
    print("Tip: pip install pyLDAvis để có visualization đẹp hơn")

# ========================
# CẤU HÌNH
# ========================
BASE_DIR      = Path("data")
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR  = BASE_DIR / "features"
MODELS_DIR    = Path("models")
PLOTS_DIR     = Path("plots")

for d in [MODELS_DIR / "lda", PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================
# CHUẨN BỊ DỮ LIỆU CHO GENSIM
# ========================

def prepare_gensim_corpus(df: pd.DataFrame):
    """
    Chuyển DataFrame sang định dạng gensim cần:
    - texts: list of list of tokens (tokenized corpus)
    - dictionary: mapping từ → ID
    - corpus: bag-of-words representation
    """
    # Tokenize lại từ processed_text
    texts = [text.split() for text in df['processed_text']]
    
    # Lọc bỏ từ quá hiếm hoặc quá phổ biến
    from gensim.models import Phrases
    
    # Tạo bigrams (từ ghép 2 từ) - quan trọng cho ESG
    bigram = Phrases(texts, min_count=3, threshold=10)
    texts_bigram = [bigram[text] for text in texts]
    
    # Tạo dictionary
    dictionary = corpora.Dictionary(texts_bigram)
    
    # Lọc: bỏ từ xuất hiện < 3 tài liệu hoặc > 85% tài liệu
    n_docs = len(texts_bigram)
    dictionary.filter_extremes(
        no_below=max(2, n_docs // 50),
        no_above=0.85,
        keep_n=10000
    )
    
    # Tạo corpus BoW
    corpus = [dictionary.doc2bow(text) for text in texts_bigram]
    
    logger.info(f"Vocabulary size: {len(dictionary):,} từ")
    logger.info(f"Corpus size: {len(corpus)} tài liệu")
    
    return texts_bigram, dictionary, corpus


# ========================
# TÌM SỐ TOPIC TỐI ƯU
# ========================

def find_optimal_topics(
    texts: list,
    dictionary,
    corpus: list,
    topic_range: range = range(3, 16)
) -> pd.DataFrame:
    """
    Thử các giá trị K khác nhau và tính Coherence Score.
    K tốt nhất = coherence cao nhất.
    
    Coherence Score đo mức độ "có nghĩa" của topic:
    - Cao hơn = topic rõ ràng, dễ hiểu hơn
    - Thang đo: thường từ -1 đến 1 (càng gần 1 càng tốt)
    """
    logger.info(f"Đang tìm K tối ưu, thử K = {list(topic_range)}...")
    
    results = []
    for k in tqdm(topic_range, desc="Testing K"):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=5,          # Số lần lặp qua corpus (tăng để chính xác hơn, chậm hơn)
            alpha='auto',      # Tự động học phân phối alpha
            eta='auto',        # Tự động học phân phối eta
            chunksize=100      # Xử lý từng batch
        )
        
        # Tính coherence (c_v là metric phổ biến nhất)
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        
        results.append({
            "num_topics": k,
            "coherence_score": round(coherence, 4)
        })
        logger.info(f"  K={k}: Coherence = {coherence:.4f}")
    
    df_results = pd.DataFrame(results)
    
    # Vẽ biểu đồ coherence
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_results['num_topics'], df_results['coherence_score'],
            marker='o', linewidth=2, markersize=8, color='#2ecc71')
    
    # Đánh dấu điểm tốt nhất
    best_k = df_results.loc[df_results['coherence_score'].idxmax(), 'num_topics']
    best_score = df_results['coherence_score'].max()
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
    ax.scatter([best_k], [best_score], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Số Topic (K)', fontsize=12)
    ax.set_ylabel('Coherence Score', fontsize=12)
    ax.set_title('Coherence Score theo Số Topic\n(chọn K tại điểm cao nhất)', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lda_coherence.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nK tối ưu: {best_k} (coherence = {best_score:.4f})")
    return df_results, best_k


# ========================
# HUẤN LUYỆN LDA
# ========================

def train_lda_model(
    corpus: list,
    dictionary,
    num_topics: int,
    passes: int = 20
) -> LdaModel:
    """
    Huấn luyện LDA với số topic đã chọn.
    passes=20 cho kết quả ổn định hơn (chậm hơn)
    """
    logger.info(f"Đang huấn luyện LDA với {num_topics} topics...")
    
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha='auto',
        eta='auto',
        chunksize=100,
        eval_every=None  # Tắt evaluation trong training để nhanh hơn
    )
    
    # Lưu model
    model_path = MODELS_DIR / "lda" / f"lda_{num_topics}topics"
    model.save(str(model_path))
    dictionary.save(str(MODELS_DIR / "lda" / "dictionary.gensim"))
    logger.info(f"Đã lưu model: {model_path}")
    
    return model


# ========================
# PHÂN TÍCH KẾT QUẢ
# ========================

def get_topic_keywords(model: LdaModel, num_words: int = 15) -> dict:
    """Lấy từ khóa chính của từng topic"""
    topics = {}
    for topic_id in range(model.num_topics):
        words = model.show_topic(topic_id, topn=num_words)
        topics[f"Topic_{topic_id}"] = {
            "keywords": [(word, round(prob, 4)) for word, prob in words],
            "top_words": [word for word, _ in words[:5]]
        }
    return topics


def label_topics_manually(topics: dict) -> dict:
    """
    Gán nhãn tên cho topic dựa trên từ khóa.
    Đây là bước MANUAL - bạn đọc từ khóa và đặt tên.
    
    Hàm này cố gắng tự đoán nhãn dựa trên ESG keywords.
    """
    esg_keywords = {
        "Environmental": {"carbon", "emission", "energy", "climate", "renewable",
                         "waste", "water", "green", "environment", "pollution"},
        "Social_Employee": {"employee", "worker", "health", "safety", "training",
                           "welfare", "labor", "workforce", "gender", "diversity"},
        "Social_Community": {"community", "society", "education", "charity",
                            "volunteer", "local", "stakeholder", "impact"},
        "Governance": {"board", "governance", "ethics", "compliance", "audit",
                      "risk", "director", "transparency", "policy", "regulation"},
        "Supply_Chain": {"supply", "chain", "supplier", "procurement", "sourcing",
                        "vendor", "material", "raw", "logistics"},
        "Financial": {"revenue", "profit", "investment", "cost", "financial",
                     "performance", "growth", "return", "capital"}
    }
    
    labeled_topics = {}
    for topic_id, topic_data in topics.items():
        topic_words = set(word for word, _ in topic_data["keywords"])
        
        # Tìm label phù hợp nhất
        best_label = topic_id
        best_overlap = 0
        for label, seed_words in esg_keywords.items():
            overlap = len(topic_words & seed_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        
        labeled_topics[topic_id] = {
            **topic_data,
            "suggested_label": best_label,
            "confidence": best_overlap
        }
    
    return labeled_topics


def get_doc_topic_distribution(model: LdaModel, corpus: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Lấy phân phối topic cho từng tài liệu.
    Mỗi tài liệu có vector topic: [0.7, 0.1, 0.2, ...]
    """
    topic_distributions = []
    
    for i, doc_bow in enumerate(corpus):
        topic_dist = model.get_document_topics(doc_bow, minimum_probability=0)
        # Chuyển thành list xác suất theo thứ tự topic
        probs = [prob for _, prob in sorted(topic_dist)]
        # Đảm bảo đủ số topic
        while len(probs) < model.num_topics:
            probs.append(0.0)
        topic_distributions.append(probs)
    
    # Tạo DataFrame
    cols = [f"topic_{i}" for i in range(model.num_topics)]
    df_topics = pd.DataFrame(topic_distributions, columns=cols)
    
    # Thêm dominant topic (topic chiếm tỷ lệ cao nhất)
    df_topics['dominant_topic'] = df_topics[cols].idxmax(axis=1)
    df_topics['dominant_prob'] = df_topics[cols].max(axis=1)
    
    # Ghép với thông tin tài liệu
    df_result = pd.concat([df[['doc_id', 'language']].reset_index(drop=True),
                           df_topics], axis=1)
    
    return df_result


# ========================
# TRỰC QUAN HÓA
# ========================

def plot_topic_keywords(topics: dict, save_path: Path):
    """Vẽ biểu đồ từ khóa cho từng topic"""
    n_topics = len(topics)
    cols = min(3, n_topics)
    rows = (n_topics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten() if n_topics > 1 else [axes]
    
    colors_list = plt.cm.Set3(np.linspace(0, 1, n_topics))
    
    for idx, (topic_id, topic_data) in enumerate(topics.items()):
        ax = axes[idx]
        keywords = topic_data["keywords"][:12]
        words, probs = zip(*keywords) if keywords else ([], [])
        
        ax.barh(words[::-1], probs[::-1], color=colors_list[idx], alpha=0.8)
        
        label = topic_data.get("suggested_label", topic_id)
        ax.set_title(f"{topic_id}\n({label})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Probability")
        ax.grid(axis='x', alpha=0.3)
    
    # Ẩn axes thừa
    for idx in range(len(topics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("LDA Topic Keywords", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Đã lưu: {save_path}")


def plot_topic_distribution(df_topics: pd.DataFrame):
    """Vẽ biểu đồ phân bố dominant topic trong corpus"""
    topic_counts = df_topics['dominant_topic'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Biểu đồ cột
    topic_counts.plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.8)
    axes[0].set_title('Số tài liệu theo Dominant Topic', fontsize=13)
    axes[0].set_xlabel('Topic')
    axes[0].set_ylabel('Số tài liệu')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Biểu đồ tròn
    topic_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                     startangle=90, colormap='Set3')
    axes[1].set_title('Tỷ lệ Dominant Topic (%)', fontsize=13)
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lda_topic_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Đã lưu: plots/lda_topic_distribution.png")


def create_pyldavis(model, corpus, dictionary):
    """Tạo interactive visualization với pyLDAvis"""
    if not HAS_PYLDAVIS:
        return
    
    try:
        vis_data = pyLDAvis.gensim_models.prepare(model, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis_data, str(PLOTS_DIR / "lda_visualization.html"))
        logger.info("  Đã lưu: plots/lda_visualization.html (mở bằng browser)")
    except Exception as e:
        logger.warning(f"  Không tạo được pyLDAvis: {e}")


# ========================
# PIPELINE CHÍNH
# ========================

def run_lda_pipeline(num_topics: int = None):
    """
    Pipeline LDA đầy đủ.
    
    Args:
        num_topics: Số topic. Nếu None → tự tìm optimal K
    """
    # 1. Load corpus
    df = pd.read_csv(PROCESSED_DIR / "corpus.csv").dropna(subset=['processed_text'])
    logger.info(f"Đã load {len(df)} tài liệu")
    
    # 2. Chuẩn bị cho gensim
    texts, dictionary, corpus = prepare_gensim_corpus(df)
    
    # 3. Tìm số topic tối ưu (nếu chưa chỉ định)
    if num_topics is None:
        logger.info("\nBước 3a: Tìm K tối ưu (mất 5-15 phút)...")
        df_coherence, num_topics = find_optimal_topics(
            texts, dictionary, corpus,
            topic_range=range(3, 12)  # Thử K từ 3 đến 11
        )
        df_coherence.to_csv(FEATURES_DIR / "lda_coherence_scores.csv", index=False)
    
    logger.info(f"\nSử dụng K = {num_topics} topics")
    
    # 4. Train model
    model = train_lda_model(corpus, dictionary, num_topics, passes=20)
    
    # 5. Lấy kết quả
    topics = get_topic_keywords(model, num_words=15)
    topics = label_topics_manually(topics)
    
    # In ra màn hình
    print("\n" + "="*50)
    print("TOPICS TÌM ĐƯỢC:")
    print("="*50)
    for topic_id, data in topics.items():
        print(f"\n{topic_id} [{data['suggested_label']}]:")
        print(f"  Top words: {', '.join(data['top_words'])}")
    
    # 6. Phân phối topic theo tài liệu
    df_doc_topics = get_doc_topic_distribution(model, corpus, df)
    df_doc_topics.to_csv(FEATURES_DIR / "doc_topic_distribution.csv", index=False)
    
    # Lưu topic info
    with open(FEATURES_DIR / "topic_info.json", 'w', encoding='utf-8') as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    
    # 7. Visualization
    logger.info("\nĐang vẽ biểu đồ...")
    plot_topic_keywords(topics, PLOTS_DIR / "lda_topic_keywords.png")
    plot_topic_distribution(df_doc_topics)
    create_pyldavis(model, corpus, dictionary)
    
    logger.info(f"""
=== KẾT QUẢ LDA ===
Số topics: {num_topics}
Files đã lưu:
  - models/lda/lda_{num_topics}topics.*
  - data/features/doc_topic_distribution.csv
  - data/features/topic_info.json
  - plots/lda_topic_keywords.png
  - plots/lda_topic_distribution.png
  - plots/lda_visualization.html (nếu có pyLDAvis)
    """)
    
    return model, topics, df_doc_topics


if __name__ == "__main__":
    print("=" * 60)
    print("BƯỚC 4: LDA TOPIC MODELING")
    print("=" * 60)
    
    # Bạn có thể chỉ định num_topics=7 nếu đã biết muốn bao nhiêu topic
    # Hoặc để None để tự tìm optimal K
    run_lda_pipeline(num_topics=None)
