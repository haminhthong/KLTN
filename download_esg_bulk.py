import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import pandas as pd
import urllib.parse
import re

BASE_DIR = Path("data/raw_pdf/EN")
BASE_DIR.mkdir(parents=True, exist_ok=True)
META_FILE = Path("data/metadata.csv")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

def search_pdf_duckduckgo(query):
    """Tìm kiếm link PDF thông qua DuckDuckGo HTML"""
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        # Lấy tất cả link kết quả
        for a in soup.find_all('a'):
            href = a.get('href', '')
            if 'uddg=' in href:
                try:
                    # Parse link thật từ DuckDuckGo redirect url
                    parsed_link = urllib.parse.unquote(href.split('uddg=')[1].split('&')[0])
                    
                    if parsed_link.lower().endswith('.pdf'):
                        # Loại trừ các file summary, brochure, highlights tự động theo file 00_
                        lower_link = parsed_link.lower()
                        if not any(word in lower_link for word in ['summary', 'highlight', 'brochure']):
                            return parsed_link
                except:
                    pass
    except Exception as e:
        print(f"  [!] Lỗi tìm kiếm: {e}")
    return None

def download_file(url, filepath):
    try:
        r = requests.get(url, headers=headers, timeout=60, stream=True)
        if r.status_code == 200:
            # Kiểm tra nội dung có phải là PDF
            content_type = r.headers.get('content-type', '')
            if 'application/pdf' not in content_type.lower() and not filepath.name.lower().endswith('.pdf'):
                print("  [!] Không phải là định dạng PDF hợp lệ.")
                return False

            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
             print(f"  [!] HTTP {r.status_code}")
    except Exception as e:
        print(f"  [!] Lỗi HTTP tải file: {e}")
    return False

def scrape_esg_reports():
    print("=== CÀO BÁO CÁO ESG THEO HƯỚNG DẪN 00_ ===")
    
    if not META_FILE.exists():
        print(f"Không tìm thấy file {META_FILE}.")
        print("Vui lòng tạo danh sách công ty ứng viên với các cột như sau: company_name, ISIN, report_year")
        return
        
    df = pd.read_csv(META_FILE)
    if not all(col in df.columns for col in ['company_name', 'report_year']):
        print("File metadata.csv cần phải có ít nhất hai cột 'company_name' và 'report_year'")
        return

    print(f"Phát hiện danh sách {len(df)} công ty.")
    downloaded = 0
    
    # Khởi tạo các cột track trạng thái tải file nếu chưa có (theo section 4.2 của file 00_)
    for col in ['path_to_pdf', 'download_date', 'file_status', 'source']:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        company = row['company_name']
        year = row['report_year']
        isin = row.get('ISIN', f'UNKNOWN_{idx}')
        
        # Bỏ qua nếu đã tải thành công trước đó
        if row.get('file_status') == 'found' and pd.notna(row.get('path_to_pdf')):
            path = Path(str(row['path_to_pdf']))
            if path.exists():
                print(f"[{idx+1}/{len(df)}] Bỏ qua {company} {year} - Đã tải sẵn.")
                continue

        # Xây dựng tên file theo chuẩn: [Tên công ty]_[ISIN]_[Năm tài chính].pdf
        safe_company = re.sub(r'[^A-Za-z0-9]', '', str(company))
        filename = f"{safe_company}_{isin}_{year}.pdf"
        filepath = BASE_DIR / filename
        
        # Cú pháp tìm kiếm (phần 4)
        query = f'site:sustainabilityreports.com OR "{company}" {year} "sustainability report" OR "ESG report" filetype:pdf'
        print(f"[{idx+1}/{len(df)}] Đang tìm: {company} - Năm {year}")
        
        pdf_url = search_pdf_duckduckgo(query)
        
        if pdf_url:
            print(f"  -> Link PDF: {pdf_url}")
            success = download_file(pdf_url, filepath)
            
            if success:
                df.at[idx, 'path_to_pdf'] = str(filepath.absolute())
                df.at[idx, 'download_date'] = time.strftime('%d/%m/%Y')
                df.at[idx, 'file_status'] = 'found'
                df.at[idx, 'source'] = 'Web Search'
                downloaded += 1
                print("  -> ✓ Tải thành công")
            else:
                df.at[idx, 'file_status'] = 'error'
                print("  -> ❌ Lỗi trong quá trình tải")
        else:
            df.at[idx, 'file_status'] = 'not_found'
            print("  -> ❌ Không tìm thấy link PDF phù hợp")
            
        time.sleep(3) # Nghỉ giữa các request để tránh bị chặn
        
        # Cập nhật kết quả vào database sau mỗi công ty
        df.to_csv(META_FILE, index=False)
        
    print(f"\nHOÀN TẤT! Đã tải mới được {downloaded} báo cáo PDF.")

if __name__ == "__main__":
    scrape_esg_reports()