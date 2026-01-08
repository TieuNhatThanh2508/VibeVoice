"""
File operations utilities
"""

import os
import re
import shutil
import uuid

from demo.colab_modules import config


def drive_save(file_copy):
    """
    Save file to Google Drive if running on Colab
    
    Args:
        file_copy: Path to file to save
    
    Returns:
        str: Path where file was saved, or None if not on Colab
    """
    drive_path = config.file.drive_path
    save_folder = os.path.join(drive_path, config.file.drive_save_folder)

    if os.path.exists(drive_path):
        print("Running on Google Colab and auto-saving to Google Drive...")
        os.makedirs(save_folder, exist_ok=True)
        dest_path = os.path.join(save_folder, os.path.basename(file_copy))
        shutil.copy2(file_copy, dest_path)
        print(f"File saved to: {dest_path}")
        return dest_path
    else:
        print("Not running on Google Colab (or Google Drive not mounted). Skipping auto-save.")
        return None


def generate_file_name(text):
    """
    Generate a unique filename from text content
    
    Args:
        text: Text content to generate name from
    
    Returns:
        str: Full path to generated filename
    """
    output_dir = config.file.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean text
    cleaned = re.sub(r"^\s*speaker\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    short = cleaned[:config.file.filename_max_length].strip()
    short = re.sub(config.file.filename_clean_pattern, '', short)
    short = short.lower().strip().replace(" ", "_")
    
    if not short:
        short = "podcast_output"
    
    unique_name = f"{short}_{uuid.uuid4().hex[:6]}"
    return os.path.join(output_dir, unique_name)

