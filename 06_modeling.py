"""
BƯỚC 6: MÔ HÌNH HÓA - RANDOM FOREST + RIDGE REGRESSION
=========================================================
Hai mô hình chính:

A) RANDOM FOREST CLASSIFICATION
   Mục tiêu: Phân loại mức độ công bố ESG: Low / Medium / High
   Input: TF-IDF features + Topic features + Sentiment features
   Output: Nhãn phân loại

   LÝ THUYẾT:
   Random Forest = tập hợp nhiều Decision Tree
   - Mỗi tree học trên subset ngẫu nhiên của data
   - Kết quả cuối = majority vote của tất cả tree
   - Ưu điểm: Không cần scale, chịu outlier tốt, cho biết feature importance

B) RIDGE REGRESSION
   Mục tiêu: Dự đoán chỉ số ESG score (liên tục)
   Input: TF-IDF features + Topic features + Sentiment
   Output: Số thực (ESG score)

   LÝ THUYẾT:
   Ridge = Linear Regression + L2 regularization
   Loss = MSE + α × Σ(βᵢ²)
   - Penalty ngăn overfitting
   - α (alpha) càng lớn → regularization càng mạnh
   - Tốt khi có nhiều features tương quan nhau (multicollinearity)

Cài thư viện:
    pip install scikit-learn pandas numpy matplotlib seaborn joblib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import logging
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_regression

# ========================
# CẤU HÌNH
# ========================
BASE_DIR      = Path("data")
FEATURES_DIR  = BASE_DIR / "features"
MODELS_DIR    = Path("models")
PLOTS_DIR     = Path("plots")
RESULTS_DIR   = Path("results")

for d in [MODELS_DIR / "ml", PLOTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ========================
# LOAD & KẾT HỢP FEATURES
# ========================

def load_all_features() -> tuple:
    """
    Load và kết hợp tất cả features:
    - TF-IDF matrix (thưa/sparse)
    - Topic distribution (từ LDA)
    - Sentiment scores
    
    Returns: X (features), doc_ids, df_combined
    """
    logger.info("Đang load features...")
    
    # 1. TF-IDF matrix
    tfidf_path = FEATURES_DIR / "tfidf_matrix.npz"
    if not tfidf_path.exists():
        raise FileNotFoundError("Chạy bước 3 (TF-IDF) trước!")
    tfidf_matrix = load_npz(tfidf_path)
    
    # 2. Doc index
    df_index = pd.read_csv(FEATURES_DIR / "doc_index.csv")
    doc_ids = df_index['doc_id'].tolist()
    
    # 3. Topic distribution
    topic_path = FEATURES_DIR / "doc_topic_distribution.csv"
    if topic_path.exists():
        df_topics = pd.read_csv(topic_path)
        topic_cols = [c for c in df_topics.columns if c.startswith('topic_')]
        
        # Align với doc_index
        df_topics = df_topics.set_index('doc_id').reindex(doc_ids).fillna(0)
        topic_features = csr_matrix(df_topics[topic_cols].values)
        logger.info(f"  Topic features: {topic_features.shape}")
    else:
        topic_features = csr_matrix((len(doc_ids), 0))
        logger.warning("  Không có topic features, bỏ qua")
    
    # 4. Sentiment features
    sentiment_path = FEATURES_DIR / "sentiment_scores.csv"
    if sentiment_path.exists():
        df_sent = pd.read_csv(sentiment_path)
        df_sent = df_sent.set_index('doc_id').reindex(doc_ids).fillna(0)
        
        sent_cols = [c for c in df_sent.columns
                    if c in ['vader_compound', 'vader_pos', 'vader_neg', 'tb_polarity', 'finbert_compound']]
        
        if sent_cols:
            sent_features = csr_matrix(df_sent[sent_cols].values)
            logger.info(f"  Sentiment features: {sent_features.shape}")
        else:
            sent_features = csr_matrix((len(doc_ids), 0))
    else:
        sent_features = csr_matrix((len(doc_ids), 0))
        logger.warning("  Không có sentiment features, bỏ qua")
    
    # 5. Kết hợp tất cả features
    feature_parts = [tfidf_matrix]
    if topic_features.shape[1] > 0:
        feature_parts.append(topic_features)
    if sent_features.shape[1] > 0:
        feature_parts.append(sent_features)
    
    X = hstack(feature_parts)
    logger.info(f"Feature matrix tổng: {X.shape}")
    
    # DataFrame thông tin
    df_combined = df_index.copy()
    if topic_path.exists() and topic_cols:
        df_combined = df_combined.merge(df_topics[topic_cols].reset_index(), on='doc_id', how='left')
    if sentiment_path.exists() and sent_cols:
        df_combined = df_combined.merge(df_sent[sent_cols].reset_index(), on='doc_id', how='left')
    
    return X, doc_ids, df_combined


def create_esg_labels(df_combined: pd.DataFrame, doc_ids: list) -> tuple:
    """
    Tạo nhãn phân loại ESG (Low/Medium/High).
    
    Chiến lược gán nhãn:
    1. Nếu có điểm ESG từ nguồn bên ngoài → dùng trực tiếp
    2. Nếu không → tạo proxy label từ đặc điểm văn bản
    
    Returns: y_class (nhãn phân loại), y_score (điểm liên tục)
    """
    # Kiểm tra metadata có ESG score không
    meta_path = BASE_DIR / "metadata.csv"
    
    if meta_path.exists():
        df_meta = pd.read_csv(meta_path)
        df_meta = df_meta.set_index('doc_id') if 'doc_id' in df_meta.columns else df_meta
        
        if 'esg_score' in df_meta.columns:
            logger.info("Dùng ESG score từ metadata")
            scores = df_meta.reindex(doc_ids)['esg_score'].fillna(df_meta['esg_score'].median())
            
            # Phân loại theo percentile
            q33 = scores.quantile(0.33)
            q67 = scores.quantile(0.67)
            labels = scores.apply(lambda x: 'Low' if x < q33 else ('High' if x > q67 else 'Medium'))
            return labels.values, scores.values
    
    # Tạo proxy label từ corpus features
    logger.info("Tạo proxy ESG labels từ đặc điểm văn bản...")
    
    corpus_path = BASE_DIR / "processed" / "corpus.csv"
    df_corpus = pd.read_csv(corpus_path).set_index('doc_id')
    df_corpus = df_corpus.reindex(doc_ids)
    
    # Proxy score dựa trên:
    # 1. Độ dài văn bản (báo cáo chi tiết → score cao hơn)
    length_score = df_corpus['token_count'].fillna(0)
    length_score = (length_score - length_score.min()) / (length_score.max() - length_score.min() + 1)
    
    # 2. Sentiment score (tích cực hơn → score cao hơn)
    sent_path = FEATURES_DIR / "sentiment_scores.csv"
    if sent_path.exists():
        df_sent = pd.read_csv(sent_path).set_index('doc_id').reindex(doc_ids)
        if 'vader_compound' in df_sent.columns:
            sent_score = df_sent['vader_compound'].fillna(0)
            sent_score = (sent_score + 1) / 2  # Normalize từ [-1,1] → [0,1]
        else:
            sent_score = pd.Series(0.5, index=doc_ids)
    else:
        sent_score = pd.Series(0.5, index=doc_ids)
    
    # 3. Topic diversity (nhiều chủ đề ESG → báo cáo đầy đủ hơn)
    topic_path = FEATURES_DIR / "doc_topic_distribution.csv"
    if topic_path.exists():
        df_topics = pd.read_csv(topic_path).set_index('doc_id').reindex(doc_ids)
        topic_cols = [c for c in df_topics.columns if c.startswith('topic_')]
        if topic_cols:
            # Entropy của topic distribution = đa dạng topic
            probs = df_topics[topic_cols].fillna(0).values
            probs = np.clip(probs, 1e-10, 1)
            entropy = -np.sum(probs * np.log(probs), axis=1)
            entropy_score = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1)
        else:
            entropy_score = np.zeros(len(doc_ids))
    else:
        entropy_score = np.zeros(len(doc_ids))
    
    # Kết hợp proxy score
    proxy_score = (
        0.4 * length_score.values +
        0.3 * sent_score.values +
        0.3 * entropy_score
    )
    
    # Phân loại theo percentile
    q33, q67 = np.percentile(proxy_score, [33, 67])
    labels = np.array(['Low' if s < q33 else ('High' if s > q67 else 'Medium')
                       for s in proxy_score])
    
    # In phân bố
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Phân bố nhãn ESG:")
    for u, c in zip(unique, counts):
        logger.info(f"  {u}: {c} ({c/len(labels)*100:.1f}%)")
    
    return labels, proxy_score


# ========================
# RANDOM FOREST CLASSIFIER
# ========================

def train_random_forest(X, y, feature_names_hint: list = None):
    """
    Huấn luyện và đánh giá Random Forest.
    """
    logger.info("\n=== RANDOM FOREST CLASSIFICATION ===")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # ---- Giảm chiều đặc trưng (quan trọng khi có nhiều features) ----
    # Chọn top K features tốt nhất
    logger.info("Đang chọn top features...")
    selector = SelectKBest(chi2, k=min(1000, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    
    # ---- Huấn luyện ----
    logger.info("Đang huấn luyện Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,    # Số cây
        max_depth=20,        # Độ sâu tối đa
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Xử lý imbalanced classes
        random_state=42,
        n_jobs=-1            # Dùng tất cả CPU cores
    )
    rf.fit(X_train_sel, y_train)
    
    # ---- Đánh giá ----
    y_pred = rf.predict(X_test_sel)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"\nKết quả Random Forest:")
    logger.info(f"  Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  F1 Macro : {f1_macro:.4f}")
    logger.info(f"  F1 Weighted: {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_sel, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    logger.info(f"  5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Lưu model
    joblib.dump(rf, MODELS_DIR / "ml" / "random_forest.pkl")
    joblib.dump(selector, MODELS_DIR / "ml" / "feature_selector.pkl")
    joblib.dump(le, MODELS_DIR / "ml" / "label_encoder.pkl")
    
    # Kết quả
    results = {
        "model": "RandomForest",
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features_selected": X_train_sel.shape[1]
    }
    
    return rf, selector, le, y_test_labels, y_pred_labels, results


def plot_confusion_matrix(y_true, y_pred, labels):
    """Vẽ Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Random Forest\n(ESG Disclosure Level)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rf_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Đã lưu: plots/rf_confusion_matrix.png")


def plot_feature_importance(rf, selector, tfidf_feature_names, n_top: int = 25):
    """Vẽ Feature Importance của Random Forest"""
    try:
        # Lấy feature names sau selection
        selected_mask = selector.get_support()
        
        # Load TF-IDF feature names
        feat_names_path = FEATURES_DIR / "tfidf_feature_names.npy"
        if feat_names_path.exists():
            tfidf_names = np.load(feat_names_path, allow_pickle=True)
            all_names = list(tfidf_names) + [f"topic_{i}" for i in range(50)] + [f"sentiment_{i}" for i in range(10)]
            selected_names = [all_names[i] for i, m in enumerate(selected_mask) if m and i < len(all_names)]
        else:
            selected_names = [f"feature_{i}" for i in range(sum(selected_mask))]
        
        # Top features
        importance = rf.feature_importances_
        top_idx = importance.argsort()[-n_top:][::-1]
        top_names = [selected_names[i] if i < len(selected_names) else f"f_{i}" for i in top_idx]
        top_vals = importance[top_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, n_top))
        ax.barh(top_names[::-1], top_vals[::-1], color=colors)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {n_top} Features - Random Forest', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "rf_feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Đã lưu: plots/rf_feature_importance.png")
    except Exception as e:
        logger.warning(f"Không vẽ được feature importance: {e}")


# ========================
# RIDGE REGRESSION
# ========================

def train_ridge_regression(X, y_score):
    """
    Huấn luyện Ridge Regression để dự đoán ESG score.
    """
    logger.info("\n=== RIDGE REGRESSION ===")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )
    
    # Giảm chiều features
    selector_reg = SelectKBest(f_regression, k=min(500, X_train.shape[1]))
    X_train_sel = selector_reg.fit_transform(X_train, y_train)
    X_test_sel  = selector_reg.transform(X_test)
    
    # RidgeCV tự động tìm alpha tốt nhất
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
    ridge_cv.fit(X_train_sel, y_train)
    
    best_alpha = ridge_cv.alpha_
    logger.info(f"Alpha tốt nhất (RidgeCV): {best_alpha}")
    
    # Dự đoán
    y_pred = ridge_cv.predict(X_test_sel)
    
    # Đánh giá
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info(f"\nKết quả Ridge Regression:")
    logger.info(f"  R²   : {r2:.4f}")
    logger.info(f"  RMSE : {rmse:.4f}")
    logger.info(f"  MAE  : {mae:.4f}")
    
    # Cross-validation
    ridge = Ridge(alpha=best_alpha)
    cv_r2 = cross_val_score(ridge, X_train_sel, y_train, cv=5, scoring='r2', n_jobs=-1)
    logger.info(f"  5-Fold CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    
    # Lưu model
    joblib.dump(ridge_cv, MODELS_DIR / "ml" / "ridge_regression.pkl")
    joblib.dump(selector_reg, MODELS_DIR / "ml" / "feature_selector_reg.pkl")
    
    results = {
        "model": "Ridge Regression",
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "cv_r2_mean": round(cv_r2.mean(), 4),
        "cv_r2_std": round(cv_r2.std(), 4),
        "best_alpha": best_alpha
    }
    
    return ridge_cv, selector_reg, y_test, y_pred, results


def plot_regression_results(y_true, y_pred, r2: float):
    """Vẽ Actual vs Predicted scatter plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue', s=30, edgecolors='white', linewidth=0.5)
    
    # Đường perfect prediction
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual ESG Score', fontsize=12)
    ax.set_ylabel('Predicted ESG Score', fontsize=12)
    ax.set_title(f'Ridge Regression: Actual vs Predicted\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.6, color='coral', s=30, edgecolors='white', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted ESG Score', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residuals Plot\n(Lý tưởng: phân tán đều quanh 0)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ridge_regression_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Đã lưu: plots/ridge_regression_results.png")


# ========================
# PIPELINE CHÍNH
# ========================

def run_modeling_pipeline():
    """Pipeline đầy đủ cho cả 2 mô hình"""
    
    # 1. Load features
    X, doc_ids, df_combined = load_all_features()
    
    # 2. Tạo labels
    y_class, y_score = create_esg_labels(df_combined, doc_ids)
    
    if len(y_class) < 10:
        logger.error("Không đủ dữ liệu để train model (cần ít nhất 10 mẫu)")
        return
    
    all_results = {}
    
    # 3. Random Forest Classification
    rf, selector, le, y_test_labels, y_pred_labels, rf_results = train_random_forest(X, y_class)
    all_results["RandomForest"] = rf_results
    
    # Visualization RF
    class_labels = sorted(le.classes_)
    plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels)
    plot_feature_importance(rf, selector, None)
    
    # 4. Ridge Regression
    ridge, selector_reg, y_test_score, y_pred_score, ridge_results = train_ridge_regression(X, y_score)
    all_results["Ridge"] = ridge_results
    
    # Visualization Ridge
    plot_regression_results(y_test_score, y_pred_score, ridge_results['r2'])
    
    # 5. Lưu kết quả
    with open(RESULTS_DIR / "model_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Tóm tắt
    print("\n" + "="*55)
    print("TÓM TẮT KẾT QUẢ MÔ HÌNH")
    print("="*55)
    print(f"\n📊 RANDOM FOREST CLASSIFICATION")
    print(f"   Accuracy  : {rf_results['accuracy']:.4f} ({rf_results['accuracy']*100:.2f}%)")
    print(f"   F1 Macro  : {rf_results['f1_macro']:.4f}")
    print(f"   CV F1     : {rf_results['cv_f1_mean']:.4f} ± {rf_results['cv_f1_std']:.4f}")
    print(f"\n📈 RIDGE REGRESSION")
    print(f"   R²        : {ridge_results['r2']:.4f}")
    print(f"   RMSE      : {ridge_results['rmse']:.4f}")
    print(f"   CV R²     : {ridge_results['cv_r2_mean']:.4f} ± {ridge_results['cv_r2_std']:.4f}")
    print(f"\nFiles đã lưu:")
    print(f"  models/ml/random_forest.pkl")
    print(f"  models/ml/ridge_regression.pkl")
    print(f"  results/model_results.json")
    print(f"  plots/rf_confusion_matrix.png")
    print(f"  plots/rf_feature_importance.png")
    print(f"  plots/ridge_regression_results.png")
    
    return all_results


if __name__ == "__main__":
    print("=" * 60)
    print("BƯỚC 6: RANDOM FOREST + RIDGE REGRESSION")
    print("=" * 60)
    run_modeling_pipeline()
