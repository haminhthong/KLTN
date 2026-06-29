


---

## 4. Chi tiết các bước trong pipeline

Hệ thống được chia thành 6 quá trình xử lý:

| Bước | Tên File | Chức năng chính |
|------|-----------|------------------|
| **1** | `01_pdf_to_text.py` | Chuyển đổi báo cáo PDF sang ngôn ngữ văn bản gốc (Text Extraction). Hỗ trợ đọc bằng pdfplumber. |
| **2** | `02_preprocessing.py` | Tiền xử lý văn bản: làm sạch text, xử lý stopword, tokenization (sử dụng nltk, spacy). |
| **3** | `03_tfidf_keywords.py` | Áp dụng TF-IDF để trích xuất và đo lường trọng số các từ khóa chính. |
| **4** | `04_lda_topic_modeling.py` | Phân tích chủ đề sử dụng LDA (Gensim) để nhóm các đoạn văn bản thành cụm nội dung. |
| **5** | `05_sentiment_analysis.py` | Phân tích cảm xúc (Sentiment Analysis) sử dụng VADER hoặc tuỳ chỉnh để lấy điểm sentiment. |
| **6** | `06_modeling.py` | Áp dụng Machine Learning (Random Forest + Ridge Regression) để dự đoán hoặc phân loại dựa trên các Features trích xuất. |

---

## 5. Kết quả đầu ra

Sau khi chạy xong pipeline (`--all` hoặc từ bước 6), hệ thống sẽ cung cấp:
- **Thời gian chạy tổng**: được hiển thị trên console.
- **Biểu đồ (Plots)**: Được lưu tại thư mục `plots/` (Wordcloud, biểu đồ LDA, thống kê mô hình).
- **Kết quả mô hình**: Xem file `results/model_results.json` để đánh giá chi tiết quá trình Training/Testing.
- Các mô hình đã huấn luyện có thể đem ra tái sử dụng (như LDA model hay ML model) nằm tại thư mục `models/`.
