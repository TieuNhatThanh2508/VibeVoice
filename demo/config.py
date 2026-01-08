"""
VibeVoice Demo Configuration File

This file contains all hyperparameters and configuration settings for VibeVoice demos.
Modify values here to change default behavior across all demo scripts.
"""

import os
import torch
from typing import List, Optional

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_CONFIG = {
    # Default model path (HuggingFace model ID or local path)
    "model_path": "microsoft/VibeVoice-1.5B",
    
    # Device for inference
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Model loading settings
    "torch_dtype": "bfloat16",  # Options: "bfloat16", "float16", "float32"
    "device_map": "auto",  # Options: "cuda", "cpu", "auto" (auto will detect available device)
    "attn_implementation": None,  # Options: None, "flash_attention_2", "sdpa"
    
    # Noise scheduler configuration
    "algorithm_type": "sde-dpmsolver++",  # Diffusion algorithm
    "beta_schedule": "squaredcos_cap_v2",  # Beta schedule type
}

# Model paths mapping for easy selection
MODEL_PATHS = {
    "1.5B": "microsoft/VibeVoice-1.5B",
    "7B": "microsoft/VibeVoice-7B",
}

# ============================================================================
# Generation Parameters
# ============================================================================

GENERATION_CONFIG = {
    # DDPM inference steps (number of diffusion steps)
    "ddpm_inference_steps": 10,
    
    # CFG (Classifier-Free Guidance) scale
    "cfg_scale": 1.3,
    "cfg_scale_min": 1.0,
    "cfg_scale_max": 2.0,
    "cfg_scale_step": 0.05,
    
    # Speech rate (speed multiplier)
    "speech_rate": 1.0,
    "speech_rate_min": 0.5,
    "speech_rate_max": 2.0,
    "speech_rate_step": 0.1,
    
    # Generation config
    "do_sample": False,
    "temperature": None,  # Not used when do_sample=False
    "top_p": None,  # Not used when do_sample=False
    "top_k": None,  # Not used when do_sample=False
    
    # Other generation settings
    "refresh_negative": True,  # Refresh negative prompt during generation
    "verbose": False,  # Print generation progress
}

# ============================================================================
# Audio Configuration
# ============================================================================

AUDIO_CONFIG = {
    # Sample rate for audio processing
    "sample_rate": 24000,
    
    # Audio normalization
    "normalize_audio": True,
    "target_dB_FS": -25,  # Target decibel level for normalization
    
    # Audio output format
    "output_format": "wav",
    "output_bit_depth": 16,  # 16-bit or 32-bit
    
    # Speech rate adjustment
    "enable_speech_rate": True,  # Enable speech rate adjustment feature
}

# ============================================================================
# BGM Configuration
# ============================================================================

BGM_CONFIG = {
    # Enable/disable BGM mixing
    "enable_bgm": False,
    
    # BGM volume (0.0 to 1.0)
    "bgm_volume": 0.3,
    "bgm_volume_min": 0.0,
    "bgm_volume_max": 1.0,
    "bgm_volume_step": 0.05,
    
    # Default BGM file path (relative to demo/voices or absolute path)
    # If None, will look for BGM files in voices directory
    "default_bgm_file": None,
}

# ============================================================================
# Gradio Demo Configuration
# ============================================================================

GRADIO_CONFIG = {
    # Server settings
    "port": 7860,
    "share": False,  # Share publicly via Gradio (set to True to create public link by default)
    "server_name": "0.0.0.0",  # Server host (0.0.0.0 for all interfaces)
    "show_error": True,
    "show_api": False,
    
    # Queue settings
    "queue_max_size": 20,
    "default_concurrency_limit": 1,  # Process one request at a time
    
    # UI Defaults
    "num_speakers": 2,
    "num_speakers_min": 1,
    "num_speakers_max": 4,
    
    # Default speaker selections (will be used if available)
    "default_speakers": [
        "en-Alice_woman",
        "en-Carter_man", 
        "en-Frank_man",
        "en-Maya_woman"
    ],
    
    # Streaming settings
    "streaming_enabled": True,
    "min_yield_interval": 15,  # Yield every N seconds during streaming
    "min_chunk_size_seconds": 30,  # Minimum chunk size in seconds before yielding
    
    # UI Text
    "script_placeholder": """Enter your podcast script here. You can format it as:

Speaker 1: Welcome to our podcast today!
Speaker 2: Thanks for having me. I'm excited to discuss...

Or paste text directly and it will auto-assign speakers.""",
    
    # Example scripts settings
    "load_examples": True,
    "max_example_duration_minutes": 15,  # Skip examples longer than this
}

# ============================================================================
# Inference Script Configuration
# ============================================================================

INFERENCE_CONFIG = {
    # Default input/output paths
    "default_txt_path": "demo/text_examples/1p_abs.txt",
    "default_output_dir": "./outputs",
    
    # Default speaker names
    "default_speaker_names": ["Andrew"],
    
    # File processing
    "auto_create_output_dir": True,
}

# ============================================================================
# Paths Configuration
# ============================================================================

PATHS_CONFIG = {
    # Directory paths (relative to demo/ folder)
    "voices_dir": "demo/voices",
    "text_examples_dir": "demo/text_examples",
    "outputs_dir": "./outputs",
}

# ============================================================================
# Advanced Settings
# ============================================================================

ADVANCED_CONFIG = {
    # Seed for reproducibility
    "seed": 42,
    
    # Logging
    "log_level": "INFO",  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    
    # Performance
    "enable_flash_attention": False,  # Enable flash attention (requires compatible GPU)
    "enable_torch_compile": False,  # Enable torch.compile for faster inference
    
    # Memory optimization
    "low_mem_mode": False,  # Use memory-efficient mode
    "gradient_checkpointing": False,  # Enable gradient checkpointing (for training)
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_model_path() -> str:
    """Get model path from config or environment variable."""
    return os.getenv("VIBEVOICE_MODEL_PATH", MODEL_CONFIG["model_path"])

def get_device() -> str:
    """Get device from config or auto-detect."""
    device = os.getenv("VIBEVOICE_DEVICE", MODEL_CONFIG["device"])
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_torch_dtype():
    """Get torch dtype from config."""
    dtype_str = MODEL_CONFIG["torch_dtype"]
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

def validate_config():
    """Validate configuration values."""
    errors = []
    
    # Validate generation config
    if not (GENERATION_CONFIG["cfg_scale_min"] <= GENERATION_CONFIG["cfg_scale"] <= GENERATION_CONFIG["cfg_scale_max"]):
        errors.append(f"cfg_scale ({GENERATION_CONFIG['cfg_scale']}) must be between min and max")
    
    if not (GENERATION_CONFIG["speech_rate_min"] <= GENERATION_CONFIG["speech_rate"] <= GENERATION_CONFIG["speech_rate_max"]):
        errors.append(f"speech_rate ({GENERATION_CONFIG['speech_rate']}) must be between min and max")
    
    if GENERATION_CONFIG["ddpm_inference_steps"] < 1:
        errors.append("ddpm_inference_steps must be >= 1")
    
    # Validate audio config
    if AUDIO_CONFIG["sample_rate"] <= 0:
        errors.append("sample_rate must be > 0")
    
    # Validate gradio config
    if not (GRADIO_CONFIG["num_speakers_min"] <= GRADIO_CONFIG["num_speakers"] <= GRADIO_CONFIG["num_speakers_max"]):
        errors.append(f"num_speakers ({GRADIO_CONFIG['num_speakers']}) must be between min and max")
    
    if GRADIO_CONFIG["port"] < 1 or GRADIO_CONFIG["port"] > 65535:
        errors.append(f"port ({GRADIO_CONFIG['port']}) must be between 1 and 65535")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# ============================================================================
# Export all configs
# ============================================================================

__all__ = [
    "MODEL_CONFIG",
    "MODEL_PATHS",
    "GENERATION_CONFIG", 
    "AUDIO_CONFIG",
    "BGM_CONFIG",
    "GRADIO_CONFIG",
    "INFERENCE_CONFIG",
    "PATHS_CONFIG",
    "ADVANCED_CONFIG",
    "get_model_path",
    "get_device",
    "get_torch_dtype",
    "validate_config",
]

