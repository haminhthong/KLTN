import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import os

BASE_DIR = Path("data/raw_pdf/EN")
BASE_DIR.mkdir(parents=True, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def download_gri_new(max_files=300):
    print("Đang crawl GRI Database (phiên bản mới)...")
    url = "https://www.globalreporting.org/search/"
    downloaded = 0
    page = 1

    while downloaded < max_files and page <= 8:   # giới hạn 8 trang
        params = {
            "report_type": "GRI Standards",
            "year_from": "2020",
            "year_to": "2025",
            "page": page
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"Lỗi trang {page}: {response.status_code}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = soup.find_all('a', href=lambda x: x and x.endswith('.pdf'))
        
        print(f"Trang {page}: tìm thấy {len(pdf_links)} link PDF")
        
        for link in pdf_links:
            pdf_url = link['href']
            if not pdf_url.startswith('http'):
                pdf_url = 'https://www.globalreporting.org' + pdf_url
            
            filename = pdf_url.split('/')[-1]
            filepath = BASE_DIR / filename
            
            if filepath.exists():
                continue
                
            print(f"Đang tải: {filename}")
            try:
                r = requests.get(pdf_url, headers=headers, timeout=60)
                if r.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(r.content)
                    downloaded += 1
                    print(f"✓ {downloaded}/{max_files}")
                    time.sleep(2)  # tránh bị chặn
            except Exception as e:
                print(f"❌ Lỗi: {filename}")
            
            if downloaded >= max_files:
                break
                
        page += 1
        time.sleep(3)

    print(f"\nHOÀN TẤT! Đã tải {downloaded} file PDF vào data/raw_pdf/EN/")

if __name__ == "__main__":
    download_gri_new(max_files=300)