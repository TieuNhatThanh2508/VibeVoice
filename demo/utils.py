"""
Utility functions for VibeVoice Gradio Demo
"""

import numpy as np
import torch


def convert_to_16_bit_wav(data):
    """
    Convert audio data to 16-bit WAV format.
    
    Args:
        data: Audio data (numpy array or torch tensor)
        
    Returns:
        16-bit integer audio data
    """
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data

