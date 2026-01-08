"""
Voice management for VibeVoice Colab Demo
"""

import os
import re
from typing import Dict, List
from .colab_config import config


class VoiceManager:
    """Manages voice presets and speaker selection"""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize voice manager
        
        Args:
            base_dir: Base directory for demo files, defaults to demo/
        """
        if base_dir is None:
            # Get demo/ directory (parent of colab_modules/)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.voices_dir = os.path.join(base_dir, config.file.voices_dir)
        self.voice_presets: Dict[str, str] = {}
        self.available_voices: Dict[str, str] = {}
        
        self.setup_voice_presets()
    
    def setup_voice_presets(self):
        """Scan and load voice presets from voices directory"""
        if not os.path.exists(self.voices_dir):
            print(f"Warning: Voices directory not found at {self.voices_dir}, creating it.")
            os.makedirs(self.voices_dir, exist_ok=True)
        
        self.voice_presets = {}
        audio_files = [
            f for f in os.listdir(self.voices_dir)
            if f.lower().endswith(tuple(config.audio.supported_formats))
            and os.path.isfile(os.path.join(self.voices_dir, f))
        ]
        
        for audio_file in audio_files:
            name = os.path.splitext(audio_file)[0]
            self.voice_presets[name] = os.path.join(self.voices_dir, audio_file)
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            print("Warning: No voice presets found.")
        else:
            print(f"Found {len(self.available_voices)} voice files in {self.voices_dir}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice names"""
        return list(self.available_voices.keys())
    
    def get_voice_path(self, voice_name: str) -> str:
        """Get file path for a voice name"""
        return self.available_voices.get(voice_name)
    
    def add_voice_file(self, file_path: str) -> str:
        """
        Add a voice file to the voices directory
        
        Args:
            file_path: Path to voice file to add
        
        Returns:
            str: Name of the added voice
        """
        import shutil
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.voices_dir, filename)
        shutil.copy(file_path, dest_path)
        
        # Refresh presets
        self.setup_voice_presets()
        
        return os.path.splitext(filename)[0]
    
    @staticmethod
    def get_num_speakers_from_script(script: str) -> int:
        """
        Determine number of speakers from script
        
        Args:
            script: Script text with speaker labels
        
        Returns:
            int: Number of unique speakers found
        """
        speakers = set(re.findall(r'^Speaker\s+(\d+)\s*:', script, re.MULTILINE | re.IGNORECASE))
        return max(int(s) for s in speakers) if speakers else 1

