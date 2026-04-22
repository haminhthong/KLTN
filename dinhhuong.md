# ĐỊNH HƯỚNG CẢI TIẾN CODE TỪNG PHẦN (TIẾN TỚI BÁO CÁO KHOA HỌC)

Tài liệu này vạch ra lộ trình nâng cấp trực tiếp vào từng file source code cụ thể trong thư mục. Để có thể đưa đề tài này thành một bài báo khoa học (Scientific Paper) chất lượng, chúng ta cần thay đổi mã nguồn từ mức "chạy được" sang mức "tối ưu và có phương pháp luận rõ ràng để chứng minh".

---

## 1. `01_pdf_to_text.py` - Từ Text thô sang Text cấu trúc (Layout-Aware)
*   **Hạn chế hiện tại:** Chỉ lấy chữ từ PDF nối vào nhau. Báo cáo ESG thường chứa bảng biểu (Scope 1, 2, 3 GHG Emissions) và chia 2-3 cột. Đọc thô sẽ làm rối logic câu.
*   **Nâng cấp code:**
    *   **Thêm logic Layout-aware:** Sử dụng thư viện nâng cao như `pdfplumber` (kết hợp thuộc tính bốc tách bounding-box) hoặc model deep learning `LayoutLMv3` để phân biệt được đâu là "Đoạn văn", đâu là "Tiêu đề", và đâu là "Chân trang" (Footer/Header/Disclaimers).
    *   **Bốc tách riêng Bảng (Tables):** Tích hợp thêm thư viện thư viện `Camelot` hoặc `Tabula` vào file này để lấy riêng các số liệu môi trường ra thành file CSV thay vì để nó trộn lẫn vào kho khối text văn xuôi. Papers rất thích dạng feature đa phương thức (Chữ + Số bảng).

## 2. `02_preprocessing.py` - Tiền xử lý chuyên biệt (Domain-Specific)
*   **Hạn chế hiện tại:** Dùng bộ Stopwords chung chung của Tiếng Anh/Tiếng Việt.
*   **Nâng cấp code:**
    *   **ESG Stopword Dictionary:** Xây dựng phần khởi tạo 1 file `esg_stopwords.txt` bao gồm các từ sáo rỗng thường xuất hiện trong báo cáo (ví dụ: *"page", "index", "forward-looking", "copyright", "report"*). Nếu không lọc các từ này, model sẽ bị nhiễu.
    *   **Sentence Segmentation (Tách câu định danh):** Thay vì Lemmatize (Đưa về từ gốc) toàn cục phá vỡ cấu trúc câu, hãy xử lý giữ nguyên câu (Sentence) để nhường đất diễn cho các mô hình Transformer phía sau.

## 3. `03_tfidf_keywords.py` - Tiến tới Vector hóa Không gian Dày (Dense Embeddings)
*   **Hạn chế hiện tại:** TF-IDF tạo ra ma trận rất thưa thớt (Sparse), thiếu kết nối ngữ nghĩa. Xếp chữ "Warm" và "Hot" vào 2 cột không liên quan.
*   **Nâng cấp code:**
    *   Giữ nguyên code `TF-IDF` làm **Baseline Model (Mô hình đối chứng)**.
    *   **Thêm hàm `extract_dense_embeddings()`:** Import model nhúng từ thư viện `sentence-transformers` (như model `all-MiniLM-L6-v2` hoặc `FinBERT`). Biến đổi mỗi báo cáo thành 1 chuỗi Vector dày đặc 768 chiều.
    *   **Tính Novelty:** Bài báo của bạn sẽ so sánh trực tiếp *Sparse Features (TF-IDF)* vs *Dense Features (Transformer Embs)*.

## 4. `04_lda_topic_modeling.py` - Tìm Số Topic tối ưu Toán học
*   **Hạn chế hiện tại:** Số lượng topic (K) thường do dev tự gán tay cứng (ví dụ K=5, K=10).
*   **Nâng cấp code:**
    *   **Tự động hóa số Topic (Grid Search):** Thêm một vòng lặp chạy từ K=3 đến K=15. Mỗi vòng tính một hệ số **Coherence Score (C_v)**. Vẽ đồ thị đường cong và chọn K tự động tại đỉnh đồ thị.
    *   **Alternative Model:** Cài đặt thêm nhánh chạy `BERTopic` (cực kỳ hot trên các giấy biên khảo hiện nay) bên cạnh LDA cũ.

## 5. `05_sentiment_analysis.py` - Cảm xúc Độ Phân Giải Cao (Contextual Sentiment)
*   **Hạn chế hiện tại:** Tính tổng Sentiments của cả cuốn báo cáo dài 100 trang cùng lúc.
*   **Nâng cấp code:**
    *   **Aspect-Based Sentiment Analysis (ABSA):** Cắt báo cáo ra thành nghìn câu. Lọc ra những câu chứa cụm từ khóa Environment (Carbon, Water), Social (Labor, Gender), Governance (Board, Audit).
    *   Tiếp đó, tính Sentiment Score **cho riêng từng mục E, S, và G**. Bài báo khoa học sẽ hay hơn vạn lần khi bạn kết luận được: *"Doanh nghiệp khen ngợi bản thân nhiều nhất ở mảng Quản trị (G), nhưng lại có giọng văn khá tiêu cực/phòng thủ khi nhắc tới Môi trường (E)"*.

## 6. `06_modeling.py` - Xây dựng Khung Thử Nghiệm (Ablation & XAI)
*   **Hạn chế hiện tại:** Chỉ có model Random Forest và Ridge. Thiếu Ground truth và Giải thích.
*   **Nâng cấp code:**
    *   **Mapping Ground Truth:** Sửa hàm `create_esg_labels` để đọc Data Label thật lấy từ một tệp điểm S&P/Refinitiv (Bỏ cơ chế Proxy Lables).
    *   **Ablation Study Matrix:** Viết một function tự động chạy vòng lặp tắt/mở feature. 
        - Lần 1: Input = TF-IDF
        - Lần 2: Input = BERT Embeddings
        - Lần 3: Input = BERT + Topic
        - Lần 4: Input = BERT + Topic + Sentiment
      Rồi in ra 1 Bảng DataFrame so sánh 4 thông số. Xong bảng này là đã được 60% bài báo khoa học.
    *   **Thêm thư viện SHAP:** Ở đoạn sau khi train Random Forest xong, khởi tạo `shap.TreeExplainer(rf)` và xuất ra lệnh lưu file đồ thị `shap_summary_plot.png`.

## 7. `run_pipeline.py` - Hỗ trợ Tracking Báo cáo
*   **Hạn chế hiện tại:** In output ra màn hình Console, khó tracking nếu test hàng chục parameter.
*   **Nâng cấp code:**
    *   **Tích hợp MLflow hoặc Weights & Biases (W&B):** Thêm một vài dòng logging. Mỗi lần gọi file python này chạy, nó sẽ tự lưu lại tham số (K topics, bộ weights, accuracy, F1) vào 1 dashboard nội bộ để sau này bạn dễ tổng hợp xuất ra các Bảng Thống kê cho bài báo học thuật.
