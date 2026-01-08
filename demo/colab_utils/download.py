"""
Download utilities for model and files
"""

import os
import requests
import urllib.request
import urllib.error
from tqdm.auto import tqdm


def download_file(url, download_file_path, redownload=False):
    """
    Download a single file with progress bar
    
    Args:
        url: URL to download from
        download_file_path: Local path to save file
        redownload: Whether to redownload if file exists
    
    Returns:
        bool: True if successful, False otherwise
    """
    base_path = os.path.dirname(download_file_path)
    os.makedirs(base_path, exist_ok=True)
    
    if os.path.exists(download_file_path):
        if redownload:
            os.remove(download_file_path)
            tqdm.write(f"♻️ Redownloading: {os.path.basename(download_file_path)}")
        elif os.path.getsize(download_file_path) > 0:
            tqdm.write(f"✔️ Skipped (already exists): {os.path.basename(download_file_path)}")
            return True
    
    try:
        request = urllib.request.urlopen(url)
        total = int(request.headers.get('Content-Length', 0))
    except urllib.error.URLError as e:
        print(f"❌ Error: Unable to open URL: {url}")
        print(f"Reason: {e.reason}")
        return False
    
    with tqdm(total=total, desc=os.path.basename(download_file_path), unit='B', unit_scale=True, unit_divisor=1024) as progress:
        try:
            urllib.request.urlretrieve(
                url, 
                download_file_path,
                reporthook=lambda count, block_size, total_size: progress.update(block_size)
            )
        except urllib.error.URLError as e:
            print(f"❌ Error: Failed to download {url}")
            print(f"Reason: {e.reason}")
            return False
    
    tqdm.write(f"⬇️ Downloaded: {os.path.basename(download_file_path)}")
    return True


def download_model(repo_id, download_folder="./", redownload=False):
    """
    Download a HuggingFace model repository
    
    Args:
        repo_id: HuggingFace model repository ID (e.g., "microsoft/VibeVoice-1.5B")
        download_folder: Folder to download to
        redownload: Whether to redownload existing files
    
    Returns:
        str: Path to downloaded model directory, or None if failed
    """
    if not download_folder.strip():
        download_folder = "."
    
    url = f"https://huggingface.co/api/models/{repo_id}"
    download_dir = os.path.abspath(f"{download_folder.rstrip('/')}/{repo_id.split('/')[-1]}")
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"📂 Download directory: {download_dir}")
    
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Error:", response.status_code, response.text)
        return None
    
    data = response.json()
    siblings = data.get("siblings", [])
    files = [f["rfilename"] for f in siblings]
    
    print(f"📦 Found {len(files)} files in repo '{repo_id}'. Checking cache ...")
    
    for file in tqdm(files, desc="Processing files", unit="file"):
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
        file_path = os.path.join(download_dir, file)
        download_file(file_url, file_path, redownload=redownload)
    
    return download_dir

