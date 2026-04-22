# Hướng dẫn Thu thập 300 Báo cáo ESG

theo báo cáo tieeens độ lần 1

Cách làm:
1. Vào link trên
2. Filter: Report type = "GRI Standards"
3. Filter: Language = English / Vietnamese
4. Download PDF về máy
5. Đặt tên file theo quy tắc: COMPANY_YEAR_LANG.pdf
   Ví dụ: Unilever_2022_EN.pdf, Vinamilk_2022_VN.pdf

## Nguồn 3: Công ty Quốc tế (Tiếng Anh)
Sectors đề xuất (chọn 1-2 ngành để kết quả có ý nghĩa):

### Năng lượng:
- Shell, BP, TotalEnergies, Equinor, Chevron

### Tiêu dùng:
- Unilever, Nestle, P&G, Danone, L'Oreal

### Công nghệ:
- Apple, Microsoft, Google, Samsung, TSMC

### Tài chính:
- HSBC, JPMorgan, Goldman Sachs, Citigroup

## Nguồn 4: CDP Database
URL: https://www.cdp.net/en/responses
- Báo cáo climate change, water, forests
- Có thể download miễn phí sau khi đăng ký

## Nguồn 5: Responsible Business Exchange
URL: https://www.responsible-business.org

---

## Cấu trúc thư mục Dataset

```
data/
├── raw_pdf/
│   ├── EN/          ← Báo cáo tiếng Anh
│   │   ├── Unilever_2022_EN.pdf
│   │   ├── Apple_2022_EN.pdf
│   │   └── ...
│   └── VN/          ← Báo cáo tiếng Việt
│       ├── Vinamilk_2022_VN.pdf
│       └── ...
├── raw_text/        ← Sau khi convert PDF
├── processed/       ← Sau khi preprocessing
└── metadata.csv     ← Thông tin từng báo cáo
```

## File metadata.csv (tạo thủ công)

Cần có các cột:
```
filename, company, year, country, sector, language, source_url
Unilever_2022_EN.pdf, Unilever, 2022, UK, Consumer, EN, https://...
Vinamilk_2022_VN.pdf, Vinamilk, 2022, VN, Consumer, VN, https://...
```

## Ghi chú quan trọng:
- Nên lấy báo cáo từ 2019-2023 để có tính nhất quán
- Ưu tiên báo cáo ESG riêng biệt hơn Annual Report
- Nếu thiếu, có thể lấy Annual Report phần ESG/Sustainability
- Tối thiểu cần 200 báo cáo để model chạy có ý nghĩa
