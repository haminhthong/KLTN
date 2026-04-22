# LÝ THUYẾT CÁC PHƯƠNG PHÁP TRONG PIPELINE ESG

## 1. TF-IDF (Term Frequency – Inverse Document Frequency)

### Vấn đề TF-IDF giải quyết:
Làm sao biết từ nào THỰC SỰ quan trọng trong 1 tài liệu?
- Từ "the", "and", "company" xuất hiện nhiều → không có ý nghĩa
- Từ "carbon_emission" xuất hiện nhiều trong 1 báo cáo → quan trọng!

### Công thức:

```
TF(t, d)  = Số lần từ t xuất hiện trong tài liệu d
            ─────────────────────────────────────────
            Tổng số từ trong tài liệu d

IDF(t)    = log( Tổng số tài liệu N  )
                 ─────────────────────
                 Số tài liệu chứa từ t

TF-IDF(t,d) = TF(t,d) × IDF(t)
```

### Ví dụ cụ thể với ESG:
```
Corpus = 300 báo cáo ESG

Từ "carbon":
  - Xuất hiện 50 lần trong báo cáo Unilever (500 từ)
  - TF = 50/500 = 0.1
  - Chỉ 80 trong 300 báo cáo có từ "carbon"
  - IDF = log(300/80) = log(3.75) = 1.32
  - TF-IDF = 0.1 × 1.32 = 0.132  ← Cao! Từ quan trọng

Từ "company":
  - Xuất hiện 100 lần trong báo cáo Unilever
  - TF = 100/500 = 0.2
  - Nhưng 290 trong 300 báo cáo đều có từ "company"
  - IDF = log(300/290) = log(1.03) = 0.014
  - TF-IDF = 0.2 × 0.014 = 0.003  ← Thấp! Từ không đặc trưng
```

---

## 2. LDA (Latent Dirichlet Allocation)

### Giả định của LDA:
- Mỗi TÀI LIỆU = hỗn hợp của K chủ đề (topic)
- Mỗi CHỦ ĐỀ = phân phối xác suất trên tất cả các từ

### Ví dụ trực quan:
```
Báo cáo ESG của Unilever:
  60% Topic "Environment" (carbon, emission, renewable...)
  25% Topic "Social"      (employee, health, community...)
  15% Topic "Governance"  (board, ethics, compliance...)

Báo cáo ESG của VinGroup:
  30% Topic "Environment"
  50% Topic "Social"
  20% Topic "Governance"
```

### Quá trình học của LDA:
```
Input:  300 báo cáo ESG (sau preprocessing)
Goal:   Tìm K topic sao cho likelihood của corpus được maximize

Bước 1: Khởi tạo ngẫu nhiên: gán mỗi từ vào 1 topic ngẫu nhiên
Bước 2: Lặp nhiều lần (passes):
  Với mỗi từ w trong tài liệu d:
    - Giả vờ rút w ra khỏi topic hiện tại
    - Tính xác suất w thuộc về topic k:
      P(topic k | w, d) ∝ P(w | topic k) × P(topic k | d)
    - Gán lại w vào topic có P cao nhất
Bước 3: Sau nhiều vòng lặp, topic distribution ổn định
```

### Coherence Score:
```
Đo mức độ "có nghĩa" của 1 topic bằng cách kiểm tra:
Các từ top trong 1 topic có thường xuất hiện cùng nhau không?

Topic tốt:  carbon, emission, climate, renewable  → thường xuất hiện cùng nhau
Topic xấu:  carbon, employee, board, water        → ít liên quan

Coherence (c_v) thường từ 0.3 đến 0.7
Trên 0.5 = khá tốt
```

---

## 3. Sentiment Analysis

### VADER:
```
- Dựa trên từ điển (lexicon-based)
- Mỗi từ được gán điểm cảm xúc từ trước:
  "excellent" → +2.0
  "poor"      → -1.5
  "carbon"    → 0 (neutral)
  
- Compound score: tổng hợp tất cả → normalize về [-1, 1]
  ≥ 0.05  → Positive
  ≤ -0.05 → Negative
  Còn lại → Neutral

Ưu điểm: Nhanh, không cần training
Nhược điểm: Không hiểu ngữ cảnh
```

### FinBERT:
```
- BERT (Bidirectional Encoder Representations from Transformers)
  được fine-tune trên dữ liệu tài chính
  
- BERT đọc cả câu, hiểu ngữ cảnh hai chiều:
  "The company FAILED to reduce emissions" → Negative
  "The company DID NOT fail in sustainability" → Positive (đảo nghĩa)
  VADER sẽ nhầm câu thứ 2 là Negative vì từ "not, fail"!
  
- Fine-tuned trên: Reuters financial news, earnings calls, SEC filings
  → Hiểu văn bản tài chính/ESG tốt hơn

Ưu điểm: Chính xác hơn nhiều
Nhược điểm: Chậm hơn, cần nhiều RAM
```

---

## 4. Random Forest

### Ý tưởng cốt lõi:
```
"Đám đông thông minh hơn cá nhân"

Thay vì 1 Decision Tree dễ overfit,
Random Forest = 200 Decision Tree,
mỗi tree học trên:
  - Subset ngẫu nhiên của data (bootstrap sampling)
  - Subset ngẫu nhiên của features (feature bagging)

Prediction = Majority vote của 200 trees
```

### Decision Tree (1 cây):
```
Ví dụ phân loại ESG Low/Medium/High:

root: vader_compound > 0.1?
├── YES → topic_0_prob > 0.4?
│         ├── YES → High (nhiều chủ đề môi trường + tích cực)
│         └── NO  → Medium
└── NO  → token_count > 5000?
          ├── YES → Medium (dài nhưng không tích cực)
          └── NO  → Low
```

### Feature Importance:
```
Random Forest tự động tính được feature nào quan trọng nhất:
  - carbon_emission: 0.045  ← Từ khóa ESG quan trọng
  - topic_0_prob:    0.038  ← Topic môi trường quan trọng
  - vader_compound:  0.031  ← Sentiment quan trọng
  - the:             0.001  ← Không quan trọng
```

---

## 5. Ridge Regression

### Vấn đề với Linear Regression thông thường:
```
Khi có 5000 features (TF-IDF), nhiều features tương quan nhau
→ Hệ số β rất lớn → Overfitting

Ví dụ:
"carbon" và "emission" thường xuất hiện cùng nhau
Linear Regression không biết phân bổ weight giữa 2 từ này
→ 1 từ có weight +1000, từ kia -999 → Không ổn định
```

### Ridge giải quyết bằng L2 Regularization:
```
Hàm mất mát thông thường:
  Loss = Σ(y_i - ŷ_i)² ← Minimize sai số dự đoán

Ridge thêm Penalty:
  Loss = Σ(y_i - ŷ_i)² + α × Σ(βj²)
                           ↑
                     Phạt hệ số quá lớn

Kết quả: β buộc phải nhỏ → model ổn định hơn

Tham số α:
  α nhỏ (0.001): Ít regularization → giống Linear Regression
  α lớn (1000) : Nhiều regularization → β gần 0 → Underfitting
  α tốt nhất:    Tìm bằng Cross-Validation (RidgeCV)
```

---

## 6. Đánh giá Mô hình

### Classification (Random Forest):
```
Accuracy = Số dự đoán đúng / Tổng số dự đoán
  → 75% accuracy = đúng 75% trường hợp
  Nhược điểm: Misleading khi class imbalanced
  (VD: 90% High → luôn đoán High → accuracy 90% nhưng vô dụng)

Precision (Độ chính xác):
  Trong tất cả lần đoán "High", bao nhiêu % thực sự High?
  Precision = TP / (TP + FP)

Recall (Độ bao phủ):
  Trong tất cả báo cáo thực sự "High", model tìm được bao nhiêu %?
  Recall = TP / (TP + FN)

F1-Score = Trung bình điều hòa của Precision và Recall
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  → Cân bằng cả hai tiêu chí
  
F1 Macro = Trung bình F1 của tất cả classes (không quan tâm số lượng)
F1 Weighted = Trung bình F1 có trọng số theo số lượng mỗi class
```

### Regression (Ridge):
```
R² (R-squared / Coefficient of Determination):
  R² = 1 - SS_residual / SS_total
  R² = 1.0 → Dự đoán hoàn hảo
  R² = 0.0 → Tệ như chỉ đoán mean
  R² < 0   → Tệ hơn đoán mean!
  
  ESG: R² > 0.6 là tốt, R² > 0.8 là rất tốt

RMSE (Root Mean Squared Error):
  RMSE = √[Σ(y_i - ŷ_i)² / n]
  → Đơn vị giống với target variable
  → Penalize nặng outliers (vì bình phương)
  
MAE (Mean Absolute Error):
  MAE = Σ|y_i - ŷ_i| / n
  → Đơn vị giống target, ít nhạy với outliers hơn RMSE
```

---

## 7. Tại sao cần Combine Features?

```
TF-IDF alone:    Biết từ ngữ, không biết cảm xúc
LDA alone:       Biết chủ đề, không biết từ cụ thể
Sentiment alone: Biết cảm xúc, không biết nội dung

Kết hợp cả 3:
  Input Vector = [TF-IDF (5000 dims)] + [Topics (7 dims)] + [Sentiment (5 dims)]
                = 5012 chiều đặc trưng
  
  Random Forest có thể học được rule phức tạp:
  IF (high carbon_emission TF-IDF) AND (topic_0 > 0.5) AND (positive sentiment)
  THEN ESG_level = "High"
  
  Ridge có thể tìm linear combination:
  ESG_score = 0.3 × carbon_tfidf + 0.2 × topic_0 + 0.1 × vader_compound + ...
```
