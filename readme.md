# Hướng dẫn chạy và sử dụng ESG Pipeline

Dự án này là một hệ thống Machine Learning và Text Mining hoàn chỉnh để phân tích các Báo cáo ESG (Môi trường, Xã hội, Quản trị) bằng cả tiếng Anh và tiếng Việt. **cần check lại kỉ file này**


## Mục lục
1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Thiết lập dữ liệu](#2-thiết-lập-dữ-liệu)
3. [Cách thức chạy pipeline](#3-cách-thức-chạy-pipeline)
4. [Chi tiết các bước trong pipeline](#4-chi-tiết-các-bước-trong-pipeline)
5. [Kết quả đầu ra](#5-kết-quả-đầu-ra)

---

## 1. Cài đặt môi trường

Để chạy toàn bộ dự án, bạn cần cài đặt các thư viện Python tiên quyết.

Mở terminal/command prompt và chạy lệnh sau:
```bash
pip install -r requirements.txt
```
## 2. Thiết lập dữ liệu

### 2.1. Tạo cấu trúc thư mục



### 2.2. Chuẩn bị file PDF & Metadata
- **File PDF**: Copy tất cả các báo cáo ESG/Phát triển bền vững bằng tiếng Anh vào `data/raw_pdf/EN` và tiếng Việt vào `data/raw_pdf/VN`. *(chi tiết về cách thu thập, xem thêm file `00_dataset_guide.md`)*.
- **File Metadata**: Tạo (hoặc tải) file `metadata.csv` đặt trong thư mục `data/`. File này cung cấp thông tin như `filename`, `company`, `year`,... Nếu có thông tin về `esg_score`, mô hình sẽ dùng nó làm nhãn giám sát. Nếu thiếu, mô hình sẽ tự động tạo proxy labels.

### 2.3. Kiểm tra dữ liệu và thư viện
Sau khi chép file, bạn hãy chạy lệnh để kiểm tra độ sẵn sàng:
```bash
python run_pipeline.py --check
```

---

## 3. Cách thức chạy pipeline

Dự án cung cấp file `run_pipeline.py` giống như một orchestrator giúp bạn chạy hệ thống rất đơn giản:

- **Chạy toàn bộ quá trình** (từ bước 1 đến bước 6):
  ```bash
  python run_pipeline.py --all
  ```
- **Chỉ chạy một bước cụ thể** (Ví dụ chạy bước 3):
  ```bash
  python run_pipeline.py --step 3
  ```
- **Tiếp tục chạy từ một bước dở dang** (Ví dụ bắt đầu từ bước 3 đến hết):
  ```bash
  python run_pipeline.py --from-step 3
  ```

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
