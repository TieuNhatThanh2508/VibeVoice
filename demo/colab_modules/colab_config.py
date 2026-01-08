"""
Configuration file for VibeVoice Colab Demo
Tất cả các siêu tham số có thể điều chỉnh được định nghĩa ở đây
"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Cấu hình cho model loading và inference"""
    # Model settings
    default_model_path: str = "microsoft/VibeVoice-1.5B"
    default_inference_steps: int = 10
    default_device: str = "cuda"  # "cuda" or "cpu"
    
    # Model loading settings
    torch_dtype_cuda: str = "bfloat16"  # "bfloat16" or "float16"
    torch_dtype_cpu: str = "float32"
    use_flash_attention: bool = False  # Set to False for T4 GPU
    
    # Diffusion scheduler settings
    algorithm_type: str = "sde-dpmsolver++"
    beta_schedule: str = "squaredcos_cap_v2"
    
    # Generation settings
    default_cfg_scale: float = 1.3
    cfg_scale_min: float = 1.0
    cfg_scale_max: float = 2.0
    cfg_scale_step: float = 0.05
    
    # Generation config
    do_sample: bool = False
    refresh_negative: bool = True
    verbose: bool = False


@dataclass
class AudioConfig:
    """Cấu hình cho audio processing"""
    # Audio format
    sample_rate: int = 24000
    channels: int = 1
    audio_subtype: str = "PCM_16"
    
    # Audio file formats supported
    supported_formats: List[str] = None
    
    # Silence trimming settings
    silence_thresh: int = -45  # dB
    min_silence_len: int = 100  # ms
    keep_silence: int = 50  # ms
    
    # Speech rate (tốc độ nói)
    default_speech_rate: float = 1.0  # 1.0 = bình thường, >1.0 = nhanh hơn, <1.0 = chậm hơn
    speech_rate_min: float = 0.5  # Tối thiểu 0.5x (chậm gấp đôi)
    speech_rate_max: float = 2.0  # Tối đa 2.0x (nhanh gấp đôi)
    speech_rate_step: float = 0.1  # Bước tăng/giảm
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']


@dataclass
class UIConfig:
    """Cấu hình cho giao diện Gradio"""
    # App settings
    app_title: str = "VibeVoice AI Podcast Generator"
    tab1_title: str = "Vibe Podcasting"
    tab2_title: str = "Generate Sample Podcast Script"
    
    # Speech rate UI label
    speech_rate_label: str = "Tốc Độ Nói (Speech Rate)"
    
    # Header HTML
    header_title: str = "🎙️ Vibe Podcasting"
    header_subtitle: str = "Generate Long-form Multi-speaker AI Podcasts with VibeVoice"
    colab_link_url: str = "https://github.com/harry2141985/Google-Collab-Notebooks"
    colab_link_text: str = "🥳 Run on Google Colab"
    
    # UI Component Labels
    podcast_settings_label: str = "### 🎛️ Podcast Settings"
    speaker_selection_label: str = "### 🎭 Speaker Selection"
    script_input_label: str = "### 📝 Script Input"
    generated_output_label: str = "### 🎵 **Generated Output**"
    download_files_label: str = "📦 Download Files"
    usage_tips_label: str = "💡 Usage Tips & Examples"
    
    # Input placeholders
    script_placeholder: str = "Speaker 1: Hi everyone, I'm Alex, and welcome back.\nSpeaker 2: And I'm lisa. Thanks for tuning in."
    
    # Button labels
    random_example_btn: str = "🎲 Random Example"
    generate_btn: str = "🚀 Generate Podcast"
    stop_btn: str = "🛑 Stop Generation"
    upload_voices_btn: str = "Add Uploaded Voices to Speaker Selection"
    generate_prompt_btn: str = "Generate Prompt"
    
    # Slider settings
    num_speakers_min: int = 1
    num_speakers_max: int = 4
    num_speakers_default: int = 2
    num_speakers_step: int = 1
    
    # Textbox settings
    script_input_lines: int = 10
    prompt_output_lines: int = 25
    
    # Default speaker names
    default_speakers: List[str] = None
    
    # Accordion settings
    upload_voices_accordion_open: bool = False
    advanced_settings_accordion_open: bool = False
    download_files_accordion_open: bool = False
    usage_tips_accordion_open: bool = False
    
    # CSS Theme
    custom_css: str = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""
    theme: str = "Soft"  # Gradio theme name
    
    def __post_init__(self):
        if self.default_speakers is None:
            self.default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']


@dataclass
class PromptBuilderConfig:
    """Cấu hình cho Prompt Builder UI"""
    title: str = "🎙️ Sample Podcast Prompt Generator"
    subtitle: str = "Paste the prompt into any LLM, and customize the propmt if you want."
    
    topic_placeholder: str = "e.g., The Future of Artificial Intelligence"
    speaker_name_placeholder: str = "e.g., Speaker {i+1}"
    
    # Example prompts
    example_prompts: List[List] = None
    
    def __post_init__(self):
        if self.example_prompts is None:
            self.example_prompts = [
                ["The Ethics of Gene Editing", 2, "Dr. Evelyn Reed", "Dr. Ben Carter", "", ""],
                ["Exploring the Deep Sea", 3, "Maria", "Leo", "Samira", ""],
                ["The Future of Space Tourism", 4, "Alex", "Zara", "Kenji", "Isla"]
            ]


@dataclass
class FileConfig:
    """Cấu hình cho file paths và directories"""
    # Directories
    voices_dir: str = "voices"  # Relative to demo/
    text_examples_dir: str = "text_examples"  # Relative to demo/
    output_dir: str = "./podcast_audio"
    
    # Google Drive settings
    drive_path: str = "/content/gdrive/MyDrive"
    drive_save_folder: str = "VibeVoice_Podcast"
    
    # File naming
    filename_max_length: int = 30
    filename_clean_pattern: str = r'[^a-zA-Z0-9\s]'
    
    # Model download
    download_folder: str = "./"
    redownload: bool = False


@dataclass
class ColabConfig:
    """Main configuration class combining all configs"""
    model: ModelConfig = None
    audio: AudioConfig = None
    ui: UIConfig = None
    prompt_builder: PromptBuilderConfig = None
    file: FileConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.audio is None:
            self.audio = AudioConfig()
        if self.ui is None:
            self.ui = UIConfig()
        if self.prompt_builder is None:
            self.prompt_builder = PromptBuilderConfig()
        if self.file is None:
            self.file = FileConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Load config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            audio=AudioConfig(**config_dict.get('audio', {})),
            ui=UIConfig(**config_dict.get('ui', {})),
            prompt_builder=PromptBuilderConfig(**config_dict.get('prompt_builder', {})),
            file=FileConfig(**config_dict.get('file', {}))
        )
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'audio': self.audio.__dict__,
            'ui': self.ui.__dict__,
            'prompt_builder': self.prompt_builder.__dict__,
            'file': self.file.__dict__
        }


# Global config instance
config = ColabConfig()

