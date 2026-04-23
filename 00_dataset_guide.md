PHƯƠNG PHÁP THU THẬP DỮ LIỆU THÔ
1. Tổng quan về phương pháp thu thập dữ liệu
Đề tài thu thập hai bộ dữ liệu độc lập – văn bản báo cáo ESG và điểm số ESG từ bên thứ ba – sau đó hợp nhất thành một dataset hoàn chỉnh cho việc huấn luyện mô hình. Chiến lược chính là đi từ danh sách công ty đã có điểm ESG xác nhận trước, rồi mới đi tìm báo cáo PDF tương ứng. Cách tiếp cận này đảm bảo 100% mẫu trong dataset đều có nhãn hợp lệ, loại bỏ rủi ro việc thu thập báo cáo mà không ghép được điểm số.
300 mẫu sạch và nhất quán được xác định là mục tiêu tối thiểu đủ đại diện cho bài toán phân loại ba lớp.
Bảng 1. Tổng quan hai bộ dữ liệu trong nghiên cứu
Bộ dữ liệu
Vai trò
Định dạng
Nguồn
300 báo cáo ESG
Biến X — đặc trưng văn bản
PDF → Text
SustainabilityReports.com
Điểm số ESG
Biến y — nhãn phân loại
Số thực [0 – 5]
KnowESG / LSEG

<<<<<<< Updated upstream
theo báo cáo tieeens độ lần 1
=======
2. Bước 1 – Xây dựng danh sách công ty ứng viên
Bước đầu tiên là xây dựng danh sách các công ty ứng viên từ KnowESG. Tại đây toàn bộ điểm ESG được lọc theo ba tiêu chí cứng, áp dụng đồng thời là:
Ngành: Chỉ lấy các công ty thuộc mã GICS Energy, Materials hoặc Industrials – các ngành này có rủi ro môi trường cao hơn các ngành khác và áp lực công bố ESG lớn từ các nhà đầu tư.
Nguồn điểm: Chỉ sử dụng điểm đánh giá của LSEG trên thang 0 – 5 (phương pháp luận mới nhất của LSEG). Mặc dù KnowESG tổng hợp điểm từ nhiều tổ chức (MSCI, Sustainalytics, LSEG) nhưng đề tài chỉ lấy một nguồn duy nhất nhằm đảm bảo tính đồng nhất tuyệt đối cho biến mục tiêu.
Giai đoạn: Báo cáo công bố trong khoảng 2020 – 2024.
Khung báo cáo: Ưu tiên báo cáo tuân thủ tiêu chuẩn GRI.
Kết quả bước này là file excel chứa khoảng tầm 400 – 500 công ty ứng viên với các trường: 
Tên trường
Vai trò
company_name
Tên công ty đã được chuẩn hóa. Ví dụ: "TotalEnergies SE" thay vì "Total" hay "TotalEnergies". Trường này chỉ dùng để đọc và kiểm tra chứ không dùng làm khóa ghép vì tên công ty có thể viết khác nhau giữa các nguồn
LEI (Legal Entity Identifier)
Mã định danh pháp nhân quốc tế gồm 20 ký tự, do tổ chức GLEIF cấp. Mỗi pháp nhân kinh doanh trên toàn cầu có một LEI duy nhất và không đổi. Ví dụ LEI của TotalEnergies là 529900S21EQ1BO4ESM68. Trường này hữu ích khi cần tra thông tin pháp nhân hoặc đối chiếu với cơ sở dữ liệu tài chính quốc tế.
ISIN (International Securities Identification Number)
Mã chứng khoán quốc tế gồm 12 ký tự (2 chữ cái mã quốc gia + 9 ký tự số + 1 check digit). Ví dụ: FR0014000MR3 là ISIN của TotalEnergies niêm yết tại Pháp. Đây là khóa ghép chính trong bước hợp nhất dữ liệu cuối cùng. Việc tìm PDF trên SustainabilityReports.com vẫn dùng tên công ty, nhưng bước ghép với điểm LSEG từ KnowESG sử dụng ISIN làm khóa.
sector_gics
Mã ngành theo chuẩn GICS (Global Industry Classification Standard), nhận một trong ba giá trị: Energy, Materials, hoặc Industrials. Trường này dùng cho hai mục đích: (1) là biến đầu vào cho stratified sampling ở bước 2, và (2) là biến kiểm soát trong mô hình để tránh mô hình học sự khác biệt ngành thay vì học đặc trưng văn bản.
report_year
Năm tài chính của báo cáo, nhận giá trị từ 2020 đến 2024. Lưu ý đây là năm tài chính, không phải năm công bố – một báo cáo FY2023 có thể được công bố vào đầu năm 2024. Trường này kết hợp với ISIN tạo thành khóa ghép kép (ISIN, report_year) ở bước 6.
esg_score_lseg
Điểm ESG do LSEG chấm, thang 0 – 5, là biến mục tiêu (biến y) cho mô hình. Trường này được dùng trực tiếp cho Ridge Regression và được rời rạc hóa thành ba nhóm Thấp/Trung bình/Cao cho Random Forest. 
>>>>>>> Stashed changes

3. Bước 2 – Stratified sampling 300 công ty
Từ danh sách các ứng viên, 300 công ty được chọn theo phương pháp stratified sampling hai chiều, nhằm đảm bảo hai yêu cầu đồng thời: (1) đại diện đủ ba ngành, và (2) phân phối tương đối cân bằng giữa ba nhóm nhãn phân loại ESG. Việc này ảnh hưởng trực tiếp đến mô hình Random Forest phân loại ba lớp nếu một nhóm nhãn chiếm dưới 20% tổng mẫu.
Các ngưỡng phân nhóm dự kiến dựa trên quan sát phân phối thực tế của điểm LSEG trong danh sách các ứng viên và có thể được điều chỉnh sau khi vẽ histogram để đảm bảo không nhóm nào chiếm dưới 20%.
Cấu trúc lấy mẫu được tổ chức thành 9 ô theo công thức 3 ngành × 3 nhóm điểm, mỗi ô mục tiêu khoảng 33 – 34 mẫu:
Bảng 2. Cấu trúc stratified sampling theo ngành và nhóm điểm ESG
Ngành (GICS)
Thấp (0.0 – 2.5)
Trung bình (2.5 – 3.5)
Cao (3.5 – 5.0)
Energy
~34 mẫu
~34 mẫu
~34 mẫu
Materials
~33 mẫu
~33 mẫu
~33 mẫu
Industrials
~33 mẫu
~33 mẫu
~33 mẫu
Tổng
~100
~100
~100

<<<<<<< Updated upstream
lấy haonf toàn theo tiến độ lần 1
=======

Trong trường hợp một ô không đủ số mẫu yêu cầu sau khi lọc, sự thiếu hụt này được ghi nhận rõ ràng là limitation trong phần phân tích, thay vì bù bằng mẫu từ ô khác. Vậy là sau bước này nhóm sẽ có ~300 mẫu.
4. Bước 3 – Thu thập báo cáo PDF
4.1. Nguồn thu thập
Toàn bộ báo cáo ESG được tải từ SustainabilityReports.com – kho lưu trữ báo cáo bền vững toàn cầu với hơn 200.000 tài liệu PDF. Việc sử dụng một nguồn duy nhất đảm bảo tính đồng nhất về định dạng và khả năng trích xuất văn bản. Trong trường hợp ngoại lệ không tìm được báo cáo trên nguồn chính, trang Investor Relations chính thức của công ty được dùng làm nguồn thay thế và ghi nhận riêng trong trường source của metadata.
4.2. Quy trình tìm kiếm
Với mỗi công ty trong danh sách 300, báo cáo được tìm kiếm theo cú pháp:
[Tên công ty] + [Năm tài chính] + "sustainability report" hoặc "ESG report"
Hai tiêu chí loại lập tức mà không cần kiểm tra nội dung là:
File không tải được, bị mã hóa hoặc yêu cầu đăng nhập.
Tên file hoặc metadata ghi rõ "Summary", "Highlights" hoặc "Brochure" – chỉ lấy bản báo cáo đầy đủ.
Sau mỗi lần tải, ba trường metadata được ghi vào file Excel quản lý: path_to_pdf, download_date, và file_status (nhận một trong ba giá trị: found, not_found, error).
Thực tế thì có khoảng 15 – 20% công ty trong danh sách sẽ không tìm được báo cáo đúng năm nên danh sách các ứng viên ban đầu được xây dựng với 400 – 500 công ty để đảm bảo sau lọc vẫn đủ 300 mẫu.
5. Bước 4 – Kiểm tra chất lượng tự động
Sau khi tải xong, toàn bộ PDF được chạy qua script Python kiểm tra tự động theo ba tiêu chí định lượng.

5.1. Tiêu chí 1 – Có text layer
Sử dụng thư viện PyMuPDF (fitz) để trích xuất văn bản thô từ PDF. Nếu tổng văn bản trích xuất được dưới 1.000 ký tự, file được phân loại là scan_image – tức file ảnh không có lớp văn bản, không thể xử lý bằng text mining.
5.2. Tiêu chí 2 – Đủ độ dài
Báo cáo có số từ dưới 5.000 từ sau khi trích xuất văn bản bị phân loại là too_short và loại khỏi dataset.
5.3. Tiêu chí 3 – Đúng ngôn ngữ
Sử dụng thư viện langdetect để xác định ngôn ngữ chủ đạo của văn bản. Báo cáo không phải tiếng Anh bị phân loại là non_english và loại khỏi dataset, nhằm đảm bảo tính đồng nhất cho các mô hình NLP downstream (LDA, Sentiment Analysis, TF-IDF) vốn tối ưu cho tiếng Anh.
Kết quả của bước này là file quality_report.csv ghi nhận trạng thái từng PDF.
6. Bước 5 – Xác nhận năm tài chính
Đây là bước thủ công không thể hoàn toàn tự động hóa nhưng được tối ưu bằng cách tập trung kiểm tra vào nhóm rủi ro cao. Thay vì đọc toàn bộ 300 báo cáo thì nhóm chỉ kiểm tra trang bìa và trang đầu – nơi thường ghi rõ năm báo cáo.
Rủi ro chính của bước này là sự không đồng nhất về năm tài chính: nhiều công ty toàn cầu kết thúc năm tài chính vào tháng 3, tháng 6 hoặc tháng 9 thay vì tháng 12. Khi đó, một báo cáo gắn nhãn "FY2023" có thể thực tế bao gồm giai đoạn từ tháng 4/2023 đến tháng 3/2024 – dẫn đến lệch năm khi ghép với điểm LSEG.
Để tối ưu thời gian, danh sách các công ty có năm tài chính không kết thúc vào tháng 12 cần kiểm tra kỹ trang bìa báo cáo. Kết quả xác nhận được ghi vào trường fiscal_year_end trong file metadata.
7. Bước 6 – Ghép thành bộ dữ liệu cuối cùng
7.1. Khóa ghép
Hai bộ dữ liệu được ghép theo khóa kép ISIN + report_year. Vì ISIN là mã định danh chứng khoán quốc tế tiêu chuẩn, duy nhất cho từng công ty trên toàn cầu và không thay đổi theo thời gian.
7.2. Cấu trúc dataset cuối
Mỗi dòng trong dataset tương ứng với một báo cáo ESG của một công ty trong một năm cụ thể. Cấu trúc đầy đủ gồm các trường sau:
Bảng 3. Cấu trúc dataset huấn luyện
Trường dữ liệu
Kiểu
Nguồn
Mô tả
company_name
Text
KnowESG
Tên công ty (chuẩn hóa)
LEI
Text
KnowESG
Legal Entity Identifier – mã định danh pháp nhân
ISIN
Text
KnowESG
Mã chứng khoán quốc tế – khóa ghép chính
sector_gics
Categorical
KnowESG
Mã ngành GICS (Energy / Materials / Industrials)
report_year
Integer
Xác nhận thủ công
Năm tài chính của báo cáo (2020 – 2024)
raw_text
Text
Script
Toàn văn báo cáo sau khi trích xuất bằng thư viện PyMuPDF
fiscal_year_end
Text
Xác nhận thủ công
Ngày kết thúc năm tài chính
esg_score
Float [0 – 5]
KnowESG / LSEG
Biến y – cho Ridge Regression
esg_label
Categorical
Tính từ score
Biến y – cho Random Forest (Thấp/TB/Cao)
path_to_pdf
Text
Thu thập
Đường dẫn file báo cáo PDF
download_date
Date
Script
Ngày tải về
file_status
Categorical
Script
Trạng thái file (found/not_found/error)
word_count
Integer
Script
Số từ sau trích xuất văn bản
source
Text
Thu thập
Nguồn tải (SustainabilityReports / IR official)


7.3. Kiểm tra chất lượng dataset cuối
Trước khi đưa vào huấn luyện mô hình, bốn kiểm tra chất lượng được thực hiện:
Kiểm tra missing values: Không có ô trống trong các cột esg_score, esg_label và raw_text.
Kiểm tra trùng lặp: Không có hai dòng nào có cùng cặp (ISIN, report_year).
Kiểm tra phân phối nhãn: Ba nhóm Thấp / Trung bình / Cao không quá mất cân bằng – không nhóm nào chiếm dưới 20% tổng mẫu.
Kiểm tra năm tài chính: Xác nhận esg_score và báo cáo PDF cùng năm tài chính thông qua trường fiscal_year_end.

9. Giới hạn của quy trình thu thập
Nghiên cứu thừa nhận bị ba giới hạn chính trong quy trình thu thập dữ liệu:
Phụ thuộc một nguồn điểm: Toàn bộ điểm ESG lấy từ LSEG thông qua KnowESG. Điểm LSEG phản ánh mức độ công bố thông tin (disclosure) hơn là hiệu suất thực tế (performance). Đây là giới hạn của phần lớn nghiên cứu ESG text mining hiện tại.
Giới hạn địa lý và ngành: Mẫu tập trung vào ba ngành GICS và ưu tiên các công ty niêm yết quốc tế có báo cáo tiếng Anh. Kết quả có thể không tổng quát hóa cho các ngành khác hoặc thị trường không sử dụng tiếng Anh làm ngôn ngữ báo cáo chính.
Giới hạn về năm: Giai đoạn 2020 – 2024 được chọn để đảm bảo đủ báo cáo trên nguồn thu thập. Xu hướng ESG sau năm 2024 chưa được phản ánh trong dataset.
10. Mẫu dữ liệu chi tiết minh họa

Trường dữ liệu
Giá trị thực tế
company_name
Equinor ASA
LEI
OW6OFBNCKXC4US5C7523
ISIN
NO0010096985
sector_gics
Energy
report_year
2024
raw_text
Trích đoạn text (500 chữ đầu): Equinor 2023 Integrated annual report 2024 Annual Report Report overview 2 INTRODUCTION CONTENTS ABOUT US OUR PERFORMANCE SUSTAINABILITY STATEMENT FINANCIAL STATEMENTS ADDITIONAL INFORMATION Equinor 2024 Annual report About us An introduction to who we are, our business and our strategy. Our performance Operational, financial and sustainability performance review, including updates on our strategic progress and technological innovation. Sustainability statement Our sustainability performance rep...
fiscal_year_end
31-Dec
esg_score
4.0
esg_label
Cao
path_to_pdf
"D:\Equinor_NO0010096985_2024.pdf"
download_date
17/04/2026
file_status
Success
word_count
174723
source
SustainabilityReports

>>>>>>> Stashed changes

