"""
Audio processing utilities for VibeVoice Colab Demo
"""

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
from .colab_config import config


class AudioProcessor:
    """Handles audio reading, processing, and silence trimming"""
    
    def __init__(self):
        self.sample_rate = config.audio.sample_rate
    
    def read_audio(self, audio_path: str, target_sr: int = None) -> np.ndarray:
        """
        Read and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate, defaults to config
        
        Returns:
            np.ndarray: Audio data as numpy array
        """
        if target_sr is None:
            target_sr = self.sample_rate
        
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def trim_silence_from_numpy(
        self,
        audio_np: np.ndarray,
        sample_rate: int = None,
        silence_thresh: int = None,
        min_silence_len: int = None,
        keep_silence: int = None
    ) -> np.ndarray:
        """
        Trim silence from audio numpy array
        
        Args:
            audio_np: Audio data as numpy array
            sample_rate: Sample rate, defaults to config
            silence_thresh: Silence threshold in dB, defaults to config
            min_silence_len: Minimum silence length in ms, defaults to config
            keep_silence: Silence to keep in ms, defaults to config
        
        Returns:
            np.ndarray: Trimmed audio data
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        if silence_thresh is None:
            silence_thresh = config.audio.silence_thresh
        if min_silence_len is None:
            min_silence_len = config.audio.min_silence_len
        if keep_silence is None:
            keep_silence = config.audio.keep_silence
        
        audio_int16 = (audio_np * 32767).astype(np.int16)
        sound = AudioSegment(
            data=audio_int16.tobytes(),
            sample_width=audio_int16.dtype.itemsize,
            frame_rate=sample_rate,
            channels=1
        )
        
        audio_chunks = split_on_silence(
            sound,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        
        if not audio_chunks:
            return np.array([0.0], dtype=np.float32)
        
        combined = sum(audio_chunks)
        samples = np.array(combined.get_array_of_samples())
        return samples.astype(np.float32) / 32767.0
    
    def change_speech_rate(self, audio_np: np.ndarray, speech_rate: float, sample_rate: int = None) -> np.ndarray:
        """
        Change speech rate of audio (tốc độ nói)
        
        Args:
            audio_np: Audio data as numpy array
            speech_rate: Speech rate multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
            sample_rate: Sample rate, defaults to config
        
        Returns:
            np.ndarray: Audio with changed speech rate
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        if speech_rate == 1.0:
            # No change needed
            return audio_np
        
        try:
            # Use librosa's time_stretch to change speed without changing pitch
            # time_stretch uses phase vocoder
            audio_stretched = librosa.effects.time_stretch(audio_np, rate=speech_rate)
            return audio_stretched
        except Exception as e:
            print(f"Error changing speech rate: {e}")
            # Fallback: return original audio
            return audio_np
    
    def save_audio(self, audio_np: np.ndarray, output_path: str, sample_rate: int = None):
        """
        Save audio numpy array to file
        
        Args:
            audio_np: Audio data as numpy array
            output_path: Path to save audio file
            sample_rate: Sample rate, defaults to config
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        sf.write(
            output_path,
            audio_np,
            sample_rate,
            subtype=config.audio.audio_subtype
        )

