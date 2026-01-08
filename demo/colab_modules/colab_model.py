"""
Model loading and management for VibeVoice Colab Demo
"""

import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from .colab_config import config


class ModelManager:
    """Manages model loading and configuration"""
    
    def __init__(self, model_path: str, device: str = None, inference_steps: int = None):
        """
        Initialize model manager
        
        Args:
            model_path: Path to model directory
            device: Device to use ("cuda" or "cpu"), defaults to config
            inference_steps: Number of inference steps, defaults to config
        """
        self.model_path = model_path
        self.device = device or config.model.default_device
        self.inference_steps = inference_steps or config.model.default_inference_steps
        
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load processor and model"""
        print(f"Loading processor & model from {self.model_path}")
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        # Determine torch dtype
        if self.device == "cuda":
            torch_dtype = getattr(torch, config.model.torch_dtype_cuda)
            device_map = self.device
        else:
            torch_dtype = getattr(torch, config.model.torch_dtype_cpu)
            device_map = "cpu"
        
        # Load model
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        
        if config.model.use_flash_attention and self.device == "cuda":
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            **load_kwargs
        )
        
        self.model.eval()
        
        # Configure noise scheduler
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type=config.model.algorithm_type,
            beta_schedule=config.model.beta_schedule
        )
        
        # Set inference steps
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        # Print attention implementation
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
    
    def get_generation_config(self):
        """Get generation configuration dictionary"""
        return {
            'do_sample': config.model.do_sample,
        }

