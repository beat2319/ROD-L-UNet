import os
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
BASE_URL = "https://download.geoservice.dlr.de/supersites/files/Hawaii/"
LOGIN_URL = "https://sso.eoc.dlr.de/eoc/auth/login?service=https://download.geoservice.dlr.de/supersites/files/"
SAVE_DIR = "./spatial_data"

# Linux Server Optimization
MAX_WORKERS = 4       # Optimized for 100Mbps (4 files at ~3MB/s each)
MAX_RETRIES = 5
CHUNK_SIZE = 1024 * 1024  # 1MB chunks are efficient for Linux I/O
TIMEOUTS = (10, 600)      # (Connect, Read)

USERNAME = "beat2319"
PASSWORD = "cU2#kkm@sFk3ZH2"

def get_authenticated_session():
    """Creates a fresh authenticated session."""
    session = requests.Session()
    # Standard Linux User-Agent
    session.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'})
    try:
        response = session.get(LOGIN_URL, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        execution_token = soup.find('input', {'name': 'execution'})['value']
        login_data = {
            'username': USERNAME, 
            'password': PASSWORD, 
            'execution': execution_token, 
            '_eventId': 'submit'
        }
        post_response = session.post(LOGIN_URL, data=login_data, allow_redirects=True, timeout=30)
        if "download.geoservice.dlr.de" in post_response.url:
            return session
    except Exception as e:
        print(f"Login Error: {e}")
    return None

def download_file(file_url):
    """Worker function to handle individual file downloads."""
    file_name = file_url.split('/')[-1]
    file_path = os.path.join(SAVE_DIR, file_name)

    # Skip logic: Check if file exists and is reasonably sized (>1MB)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 1024 * 1024:
        print(f"SKIPPING: {file_name} (Already exists)")
        return

    for attempt in range(MAX_RETRIES):
        try:
            # Authenticate inside the worker to ensure thread-specific session safety
            session = get_authenticated_session()
            if not session:
                raise Exception("Authentication failed in worker thread")

            print(f"STARTING: {file_name} (Attempt {attempt+1})")
            
            with session.get(file_url, stream=True, timeout=TIMEOUTS) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            
            print(f"FINISHED: {file_name}")
            return # Success!
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            wait = (attempt + 1) * 15
            print(f"RETRYING: {file_name} in {wait}s due to error: {e}")
            time.sleep(wait)

    print(f"FAILED PERMANENTLY: {file_name}")

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("Initial login to fetch file list...")
    master_session = get_authenticated_session()
    if not master_session:
        print("Initial login failed. Check credentials.")
        return

    print("Scanning for files...")
    resp = master_session.get(BASE_URL)
    soup = BeautifulSoup(resp.text, 'html.parser')
    all_links = [urljoin(BASE_URL, a['href']) for a in soup.find_all('a') 
                 if "TSX_" in a.get('href', '') and a['href'].endswith(".tar.gz")]

    total_files = len(all_links)
    print(f"Found {total_files} files. Parallelizing over {MAX_WORKERS} threads...")

    # Parallel Execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(download_file, all_links)

    print("--- ALL PROCESSES COMPLETE ---")

if __name__ == "__main__":
    main()