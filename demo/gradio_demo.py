"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import librosa.effects
import soundfile as sf
import torch
import os
import traceback

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

# Import local modules
try:
    from .ui_styles import CUSTOM_CSS, THEME_TOGGLE_JS
    from .utils import convert_to_16_bit_wav
except ImportError:
    # Fallback for when running as script
    from ui_styles import CUSTOM_CSS, THEME_TOGGLE_JS
    from utils import convert_to_16_bit_wav

# Import configuration - support both relative and absolute imports
try:
    from .config import (
        MODEL_CONFIG,
        MODEL_PATHS,
        GENERATION_CONFIG,
        AUDIO_CONFIG,
        BGM_CONFIG,
        GRADIO_CONFIG,
        ADVANCED_CONFIG,
        get_model_path,
        get_device,
        get_torch_dtype,
        validate_config,
    )
except ImportError:
    # Fallback for when running as script (e.g., in Colab)
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from demo.config import (
        MODEL_CONFIG,
        MODEL_PATHS,
        GENERATION_CONFIG,
        AUDIO_CONFIG,
        BGM_CONFIG,
        GRADIO_CONFIG,
        ADVANCED_CONFIG,
        get_model_path,
        get_device,
        get_torch_dtype,
        validate_config,
    )

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VibeVoiceDemo:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, inference_steps: Optional[int] = None):
        """Initialize the VibeVoice demo with model loading."""
        # Use config defaults if not provided
        self.model_path = model_path or get_model_path()
        self.device = device or get_device()
        self.inference_steps = inference_steps or GENERATION_CONFIG["ddpm_inference_steps"]
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        
        # Determine current model name from model_path
        self.current_model_name = None
        for name, path in MODEL_PATHS.items():
            if path == self.model_path:
                self.current_model_name = name
                break
        if self.current_model_name is None:
            # Default to 1.5B if not found
            self.current_model_name = "1.5B"
        
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts
        self.setup_bgm_files()  # Setup BGM files
        
    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading processor & model from {self.model_path}")
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(
            self.model_path,
        )
        
        # Load model
        torch_dtype = get_torch_dtype()
        attn_impl = MODEL_CONFIG["attn_implementation"] if MODEL_CONFIG["attn_implementation"] else None
        
        # Auto-detect device and adjust settings
        device_map = MODEL_CONFIG["device_map"]
        if device_map == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device_map = "cpu"
            # Use float32 for CPU as bfloat16 may not be supported
            if torch_dtype == torch.bfloat16:
                torch_dtype = torch.float32
                print("⚠️  Using float32 instead of bfloat16 for CPU compatibility")
        
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
        self.model.eval()
        
        # Use SDE solver from config
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type=MODEL_CONFIG["algorithm_type"],
            beta_schedule=MODEL_CONFIG["beta_schedule"]
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
    
    def switch_model(self, model_name: str):
        """Switch to a different model dynamically."""
        if self.current_model_name == model_name:
            return  # Already using this model
        
        if self.is_generating:
            raise gr.Error("Cannot switch model while generation is in progress. Please stop generation first.")
        
        print(f"Switching model from {self.current_model_name} to {model_name}...")
        
        # Unload current model
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        
        # Load new model
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_PATHS.keys())}")
        
        self.model_path = MODEL_PATHS[model_name]
        self.current_model_name = model_name
        
        # Reload model and processor (load_model will handle device detection)
        self.load_model()
        print(f"Successfully switched to model: {model_name}")
    
    def setup_bgm_files(self):
        """Setup BGM files by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        self.bgm_files = []
        
        if not os.path.exists(voices_dir):
            return
        
        # Look for BGM files (files with 'bgm' in name or separate BGM directory)
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) 
                    and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Filter for BGM files (contain 'bgm' in filename)
        bgm_files = [f for f in wav_files if 'bgm' in f.lower()]
        
        for bgm_file in bgm_files:
            full_path = os.path.join(voices_dir, bgm_file)
            if os.path.exists(full_path):
                self.bgm_files.append(full_path)
        
        # If no BGM files found, use first available voice file as fallback
        if not self.bgm_files and wav_files:
            # Use first voice file as default BGM (not ideal but works)
            default_bgm = os.path.join(voices_dir, wav_files[0])
            self.bgm_files.append(default_bgm)
            print(f"Warning: No BGM files found. Using {wav_files[0]} as default BGM.")
        else:
            print(f"Found {len(self.bgm_files)} BGM file(s)")
    
    def get_bgm_file(self) -> Optional[str]:
        """Get the default BGM file path."""
        if self.bgm_files:
            return self.bgm_files[0]  # Use first available BGM file
        return BGM_CONFIG.get("default_bgm_file")
    
    def mix_bgm(self, audio: np.ndarray, bgm_file: Optional[str], bgm_volume: float, sample_rate: int) -> np.ndarray:
        """
        Mix background music with audio output.
        
        Args:
            audio: Main audio waveform as numpy array
            bgm_file: Path to BGM file (if None, uses default)
            bgm_volume: Volume of BGM (0.0 to 1.0)
            sample_rate: Sample rate of the audio
            
        Returns:
            Mixed audio with BGM
        """
        if bgm_volume <= 0.0 or bgm_file is None:
            return audio
        
        try:
            # Get BGM file if not provided
            if bgm_file is None:
                bgm_file = self.get_bgm_file()
            
            if bgm_file is None or not os.path.exists(bgm_file):
                print(f"Warning: BGM file not found: {bgm_file}. Skipping BGM mixing.")
                return audio
            
            # Load BGM
            bgm_audio = self.read_audio(bgm_file, target_sr=sample_rate)
            
            if len(bgm_audio) == 0:
                print("Warning: Failed to load BGM. Skipping BGM mixing.")
                return audio
            
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                if audio.max() > 1.0 or audio.min() < -1.0:
                    audio = audio / np.max(np.abs(audio))
            
            if bgm_audio.dtype != np.float32:
                bgm_audio = bgm_audio.astype(np.float32)
                if bgm_audio.max() > 1.0 or bgm_audio.min() < -1.0:
                    bgm_audio = bgm_audio / np.max(np.abs(bgm_audio))
            
            # Match BGM length to audio length
            audio_len = len(audio)
            bgm_len = len(bgm_audio)
            
            if bgm_len < audio_len:
                # Loop BGM if shorter
                num_loops = int(np.ceil(audio_len / bgm_len))
                bgm_audio = np.tile(bgm_audio, num_loops)
            
            # Trim BGM if longer
            if len(bgm_audio) > audio_len:
                bgm_audio = bgm_audio[:audio_len]
            
            # Mix audio with BGM
            # Normalize both to prevent clipping
            mixed_audio = audio + (bgm_audio * bgm_volume)
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val
            
            return mixed_audio.astype(np.float32)
            
        except Exception as e:
            print(f"Error mixing BGM: {e}")
            import traceback
            traceback.print_exc()
            return audio  # Return original audio on error
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .wav files to the demo/voices directory.")
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
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
    
    def preview_voice(self, speaker_name: Optional[str]) -> Optional[tuple]:
        """
        Preview a voice sample.
        
        Args:
            speaker_name: Name of the speaker to preview
            
        Returns:
            Tuple of (sample_rate, audio_data) or None if speaker not found
        """
        if not speaker_name or speaker_name not in self.available_voices:
            return None
        
        try:
            audio_path = self.available_voices[speaker_name]
            audio_data = self.read_audio(audio_path, target_sr=AUDIO_CONFIG["sample_rate"])
            
            if len(audio_data) == 0:
                return None
            
            # Convert to 16-bit for Gradio
            audio_16bit = convert_to_16_bit_wav(audio_data)
            sample_rate = AUDIO_CONFIG["sample_rate"]
            
            return (sample_rate, audio_16bit)
        except Exception as e:
            print(f"Error previewing voice {speaker_name}: {e}")
            return None
    
    def adjust_speech_rate(self, audio: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
        """
        Adjust speech rate using time-stretching (changes speed without changing pitch).
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            rate: Speech rate multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
        
        Returns:
            Time-stretched audio
        """
        if rate == 1.0:
            return audio
        
        try:
            # Use librosa's time_stretch which preserves pitch
            # rate > 1.0 makes it faster, rate < 1.0 makes it slower
            stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
            return stretched_audio
        except Exception as e:
            print(f"Error adjusting speech rate: {e}")
            return audio
    
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 model_name: str = "1.5B",
                                 cfg_scale: float = 1.3,
                                 speech_rate: float = 1.0,
                                 enable_bgm: bool = False,
                                 bgm_volume: float = 0.3,
                                 ddpm_inference_steps: int = 10,
                                 do_sample: bool = False,
                                 temperature: Optional[float] = None,
                                 top_p: Optional[float] = None,
                                 top_k: Optional[int] = None,
                                 refresh_negative: bool = True,
                                 verbose: bool = False,
                                 normalize_audio: bool = True,
                                 target_dB_FS: float = -25,
                                 seed: Optional[int] = None) -> Iterator[tuple]:
        try:
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Set seed if provided
            if seed is not None:
                set_seed(seed)
            
            # Switch model if needed
            if model_name != self.current_model_name:
                try:
                    self.switch_model(model_name)
                except gr.Error as e:
                    self.is_generating = False
                    yield None, None, f"❌ Model switch error: {str(e)}", gr.update(visible=False)
                    return
                except Exception as e:
                    self.is_generating = False
                    yield None, None, f"❌ Failed to switch model: {str(e)}", gr.update(visible=False)
                    return
            
            # Update inference steps if changed
            if ddpm_inference_steps != self.inference_steps:
                self.inference_steps = ddpm_inference_steps
                self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            # Defend against common mistake
            script = script.replace("'", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
            # Build initial log
            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={self.inference_steps}, Speech Rate={speech_rate:.2f}x\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Load voice samples
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)
            
            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Parse script to assign speaker ID's
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line already has speaker format
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation (1-based indexing)
                    speaker_id = (len(formatted_script_lines) % num_speakers) + 1
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer, speech_rate, do_sample, temperature, top_p, top_k, refresh_negative, verbose)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = AUDIO_CONFIG["sample_rate"]
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = GRADIO_CONFIG["min_yield_interval"]
            min_chunk_size = sample_rate * GRADIO_CONFIG["min_chunk_size_seconds"]
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk
                
                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Store raw audio for later speech rate adjustment
                # We'll apply speech_rate to the complete audio at the end
                audio_16bit = convert_to_16_bit_wav(audio_np)
                
                # Store for final statistics
                all_audio_chunks.append(audio_16bit)
                
                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)
                
                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True
                
                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    
                    log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                    
                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    
                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time
            
            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True  # Mark that we yielded audio
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")
            
            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                yield None, None, error_log, gr.update(visible=False)
                return
            
            if not has_yielded_audio:
                error_log = log + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                # Convert to float32 for processing
                complete_audio_float = complete_audio.astype(np.float32) / 32767.0
                
                # Apply speech rate adjustment
                if speech_rate != 1.0:
                    complete_audio_float = self.adjust_speech_rate(complete_audio_float, sample_rate, speech_rate)
                
                # Apply BGM mixing if enabled
                if enable_bgm:
                    bgm_file = self.get_bgm_file()
                    complete_audio_float = self.mix_bgm(complete_audio_float, bgm_file, bgm_volume, sample_rate)
                
                # Normalize audio if enabled
                if normalize_audio:
                    # Normalize to target dB level
                    rms = np.sqrt(np.mean(complete_audio_float**2))
                    if rms > 0:
                        target_linear = 10 ** (target_dB_FS / 20.0)
                        current_linear = rms
                        if current_linear > 0:
                            gain = target_linear / current_linear
                            complete_audio_float = complete_audio_float * gain
                            # Prevent clipping
                            max_val = np.max(np.abs(complete_audio_float))
                            if max_val > 1.0:
                                complete_audio_float = complete_audio_float / max_val
                
                # Convert back to 16-bit
                complete_audio = convert_to_16_bit_wav(complete_audio_float)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer, speech_rate=1.0, 
                               do_sample=False, temperature=None, top_p=None, top_k=None, 
                               refresh_negative=True, verbose=False):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
                
            gen_config = {
                'do_sample': do_sample,
            }
            if do_sample:
                if temperature is not None:
                    gen_config['temperature'] = temperature
                if top_p is not None:
                    gen_config['top_p'] = top_p
                if top_k is not None:
                    gen_config['top_k'] = top_k
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config=gen_config,
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=verbose,
                refresh_negative=refresh_negative,
            )
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        if not GRADIO_CONFIG["load_examples"]:
            self.example_scripts = []
            return
            
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            
            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                max_duration = GRADIO_CONFIG["max_example_duration_minutes"]
                if minutes > max_duration:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds {max_duration}-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)
                
                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        # If no speakers found, default to 1
        if not speakers:
            return 1
        
        # Return the maximum speaker ID (assuming 1-based indexing)
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            # If script uses 0-based (legacy), convert to 1-based count
            return max_speaker + 1
        else:
            # 1-based indexing, return the max speaker ID
            return max_speaker
    

def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""
    
    # Use CSS from ui_styles module
    custom_css = CUSTOM_CSS
    
    with gr.Blocks(
        title="VibeVoice - AI Podcast Generator",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as interface:
        
        # Theme toggle button
        print("[DEBUG] Creating theme toggle button and state")
        theme_state = gr.State(value="light")
        theme_toggle_btn = gr.Button(
            "🌙",
            elem_classes="theme-toggle-btn",
            size="sm"
        )
        print(f"[DEBUG] theme_toggle_btn created: {theme_toggle_btn}")
        print(f"[DEBUG] theme_state created: {theme_state}")
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Vibe Podcasting </h1>
            <p>Generating Long-form Multi-speaker AI Podcast with VibeVoice</p>
        </div>
        """)
        
        # JavaScript for theme toggle (from ui_styles module) with debug logging
        theme_js_with_debug = THEME_TOGGLE_JS.replace(
            "window.toggleVibeVoiceTheme = function(currentTheme) {",
            """window.toggleVibeVoiceTheme = function(currentTheme) {
                console.log('[DEBUG] toggleVibeVoiceTheme called with currentTheme:', currentTheme);
            """
        ).replace(
            "const newTheme = currentTheme === 'light' ? 'dark' : 'light';",
            """const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                console.log('[DEBUG] toggleVibeVoiceTheme: newTheme =', newTheme);
            """
        ).replace(
            "localStorage.setItem('vibevoice-theme', newTheme);",
            """localStorage.setItem('vibevoice-theme', newTheme);
                console.log('[DEBUG] Theme saved to localStorage:', newTheme);
            """
        )
        gr.HTML(theme_js_with_debug)
        print("[DEBUG] Theme toggle JavaScript loaded with debug logging")
        
        # Main layout with sidebar
        # Container for sidebar overlay
        sidebar_container = gr.HTML("", visible=False, elem_id="sidebar-wrapper")
        
        with gr.Row():
            # Sidebar toggle button (fixed position)
            print("[DEBUG] Creating sidebar toggle button and state")
            sidebar_visible = gr.State(value=False)
            sidebar_toggle_btn = gr.Button(
                "⚙️",
                elem_classes="sidebar-toggle-btn",
                size="sm"
            )
            print(f"[DEBUG] sidebar_toggle_btn created: {sidebar_toggle_btn}")
            print(f"[DEBUG] sidebar_visible created: {sidebar_visible}")
            
            # Sidebar with advanced settings - full height overlay
            print("[DEBUG] Creating sidebar Column component")
            with gr.Column(scale=1, visible=False, elem_classes="sidebar-container", elem_id="sidebar") as sidebar:
                print(f"[DEBUG] sidebar Column created: {sidebar}")
                gr.Markdown("### ⚙️ **Advanced Settings**")
                
                # Speech Rate
                speech_rate = gr.Slider(
                    minimum=GENERATION_CONFIG["speech_rate_min"],
                    maximum=GENERATION_CONFIG["speech_rate_max"],
                    value=GENERATION_CONFIG["speech_rate"],
                    step=GENERATION_CONFIG["speech_rate_step"],
                    label="Speech Rate",
                    info="1.0 = normal, 0.5 = half, 2.0 = double",
                    elem_classes="slider-container"
                )
                
                # BGM Settings
                enable_bgm = gr.Checkbox(
                    label="Enable Background Music",
                    value=BGM_CONFIG["enable_bgm"]
                )
                
                bgm_volume = gr.Slider(
                    minimum=BGM_CONFIG["bgm_volume_min"],
                    maximum=BGM_CONFIG["bgm_volume_max"],
                    value=BGM_CONFIG["bgm_volume"],
                    step=BGM_CONFIG["bgm_volume_step"],
                    label="BGM Volume",
                    visible=BGM_CONFIG["enable_bgm"],
                    elem_classes="slider-container"
                )
                
                # Generation Parameters
                gr.Markdown("### 🎛️ **Generation Parameters**")
                
                cfg_scale = gr.Slider(
                    minimum=GENERATION_CONFIG["cfg_scale_min"],
                    maximum=GENERATION_CONFIG["cfg_scale_max"],
                    value=GENERATION_CONFIG["cfg_scale"],
                    step=GENERATION_CONFIG["cfg_scale_step"],
                    label="CFG Scale",
                    info="Higher = more adherence to text",
                    elem_classes="slider-container"
                )
                
                ddpm_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=GENERATION_CONFIG["ddpm_inference_steps"],
                    step=1,
                    label="DDPM Inference Steps",
                    info="More steps = better quality, slower",
                    elem_classes="slider-container"
                )
                
                do_sample = gr.Checkbox(
                    label="Enable Sampling",
                    value=GENERATION_CONFIG["do_sample"]
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    visible=GENERATION_CONFIG["do_sample"],
                    elem_classes="slider-container"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P",
                    visible=GENERATION_CONFIG["do_sample"],
                    elem_classes="slider-container"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top K",
                    visible=GENERATION_CONFIG["do_sample"],
                    elem_classes="slider-container"
                )
                
                refresh_negative = gr.Checkbox(
                    label="Refresh Negative Prompt",
                    value=GENERATION_CONFIG["refresh_negative"]
                )
                
                verbose = gr.Checkbox(
                    label="Verbose Output",
                    value=GENERATION_CONFIG["verbose"]
                )
                
                # Audio Processing
                gr.Markdown("### 🎵 **Audio Processing**")
                
                normalize_audio = gr.Checkbox(
                    label="Normalize Audio",
                    value=AUDIO_CONFIG["normalize_audio"]
                )
                
                target_dB_FS = gr.Slider(
                    minimum=-40,
                    maximum=-10,
                    value=AUDIO_CONFIG["target_dB_FS"],
                    step=1,
                    label="Target dB FS",
                    visible=AUDIO_CONFIG["normalize_audio"],
                    elem_classes="slider-container"
                )
                
                # Advanced
                gr.Markdown("### 🔧 **Advanced**")
                
                seed = gr.Number(
                    label="Random Seed",
                    value=ADVANCED_CONFIG["seed"],
                    precision=0,
                    info="Set to -1 for random"
                )
            
            # Main content area
            with gr.Column(scale=3, elem_classes="main-content-area"):
                # Model Selection
                model_selection = gr.Dropdown(
                    choices=list(MODEL_PATHS.keys()),
                    value=demo_instance.current_model_name,
                    label="Model",
                    info="Select model size (1.5B = faster, 7B = better quality)"
                )
                
                # Script Input
                gr.Markdown("### 📝 **Script Input**")
                
                script_input = gr.Textbox(
                    label="Conversation Script",
                    placeholder=GRADIO_CONFIG["script_placeholder"],
                    lines=12,
                    max_lines=20,
                    elem_classes="script-input"
                )
                
                # Number of speakers
                num_speakers = gr.Slider(
                    minimum=GRADIO_CONFIG["num_speakers_min"],
                    maximum=GRADIO_CONFIG["num_speakers_max"],
                    value=GRADIO_CONFIG["num_speakers"],
                    step=1,
                    label="Number of Speakers",
                    elem_classes="slider-container"
                )
                
                # Speaker selection
                gr.Markdown("### 🎭 **Speaker Selection**")
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                # Use config default speakers, filter to only those available
                default_speakers = [
                    s for s in GRADIO_CONFIG["default_speakers"] 
                    if s in available_speaker_names
                ]
                # Fallback to first available speakers if config speakers not found
                if not default_speakers and available_speaker_names:
                    default_speakers = available_speaker_names[:4]

                speaker_selections = []
                speaker_preview_buttons = []
                speaker_preview_audios = []
                speaker_rows = []
                
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    
                    speaker_row = gr.Row(visible=(i < 2))  # Initially show only first 2 speakers
                    with speaker_row:
                        speaker = gr.Dropdown(
                            choices=available_speaker_names,
                            value=default_value,
                            label=f"Speaker {i+1}",
                            scale=4,
                            elem_classes="speaker-item"
                        )
                        preview_btn = gr.Button(
                            "🔊 Preview",
                            size="sm",
                            scale=1,
                            variant="secondary",
                            min_width=100
                        )
                    
                    # Preview audio component (separate, below the row)
                    preview_audio = gr.Audio(
                        label=f"Preview: Speaker {i+1}",
                        type="numpy",
                        visible=False,
                        show_download_button=False,
                        autoplay=True,
                        elem_id=f"preview_audio_{i}"
                    )
                    
                    speaker_selections.append(speaker)
                    speaker_preview_buttons.append(preview_btn)
                    speaker_preview_audios.append(preview_audio)
                    speaker_rows.append(speaker_row)
                
                # Button row with Random Example on the left and Generate on the right
                with gr.Row():
                    # Random example button (now on the left)
                    random_example_btn = gr.Button(
                        "🎲 Random Example",
                        size="lg",
                        variant="secondary",
                        elem_classes="random-btn",
                        scale=1  # Smaller width
                    )
                    
                    # Generate button (now on the right)
                    generate_btn = gr.Button(
                        "🚀 Generate Podcast",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2  # Wider than random button
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 Stop Generation",
                    size="lg",
                    variant="stop",
                    elem_classes="stop-btn",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Output section
                gr.Markdown("### 🎵 **Generated Podcast**")
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="Streaming Audio (Real-time)",
                    type="numpy",
                    elem_classes="audio-output",
                    streaming=True,  # Enable streaming mode
                    autoplay=True,
                    show_download_button=False,  # Explicitly show download button
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                complete_audio_output = gr.Audio(
                    label="Complete Podcast (Download after generation)",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    show_download_button=True,  # Explicitly show download button
                    visible=False  # Initially hidden, shown when audio is ready
                )
                
                gr.Markdown("""
                *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)  
                *💡 **Complete Audio**: Will appear below after generation finishes*
                """)
                
                # Generation log
                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )
        
        # Sidebar toggle function
        # Theme toggle function with debug logging
        def toggle_theme(current_theme):
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[DEBUG] toggle_theme called with current_theme: {current_theme}")
            print(f"[DEBUG] toggle_theme called with current_theme: {current_theme}")
            
            new_theme = "dark" if current_theme == "light" else "light"
            icon = "☀️" if new_theme == "dark" else "🌙"
            
            logger.info(f"[DEBUG] toggle_theme returning: new_theme={new_theme}, icon={icon}")
            print(f"[DEBUG] toggle_theme returning: new_theme={new_theme}, icon={icon}")
            
            return gr.update(value=icon), new_theme
        
        print("[DEBUG] Setting up theme_toggle_btn.click handler")
        theme_toggle_btn.click(
            fn=toggle_theme,
            inputs=[theme_state],
            outputs=[theme_toggle_btn, theme_state],
            queue=False
        ).then(
            fn=None,
            js=f"""
            (theme) => {{
                console.log('[DEBUG] Theme toggle JS called with theme:', theme);
                if (window.toggleVibeVoiceTheme) {{
                    const result = window.toggleVibeVoiceTheme(theme);
                    console.log('[DEBUG] toggleVibeVoiceTheme result:', result);
                    return [];
                }} else {{
                    console.error('[DEBUG] toggleVibeVoiceTheme function not found!');
                    return [];
                }}
            }}
            """,
            inputs=[theme_state]
        )
        print("[DEBUG] Theme toggle handler setup complete")
        
        # Sidebar toggle function with debug logging
        def toggle_sidebar(visible):
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[DEBUG] toggle_sidebar called with visible: {visible}")
            print(f"[DEBUG] toggle_sidebar called with visible: {visible}")
            
            new_visible = not visible
            logger.info(f"[DEBUG] toggle_sidebar returning: new_visible={new_visible}")
            print(f"[DEBUG] toggle_sidebar returning: new_visible={new_visible}")
            
            return gr.update(visible=new_visible), new_visible
        
        print("[DEBUG] Setting up sidebar_toggle_btn.click handler")
        print(f"[DEBUG] sidebar_toggle_btn: {sidebar_toggle_btn}")
        print(f"[DEBUG] sidebar: {sidebar}")
        print(f"[DEBUG] sidebar_visible: {sidebar_visible}")
        sidebar_toggle_btn.click(
            fn=toggle_sidebar,
            inputs=[sidebar_visible],
            outputs=[sidebar, sidebar_visible],
            queue=False
        )
        print("[DEBUG] Sidebar toggle handler setup complete")
        
        # Conditional visibility updates
        def update_bgm_volume_visibility(enable_bgm):
            return gr.update(visible=enable_bgm)
        
        enable_bgm.change(
            fn=update_bgm_volume_visibility,
            inputs=[enable_bgm],
            outputs=[bgm_volume],
            queue=False
        )
        
        def update_sampling_params_visibility(do_sample):
            return gr.update(visible=do_sample), gr.update(visible=do_sample), gr.update(visible=do_sample)
        
        do_sample.change(
            fn=update_sampling_params_visibility,
            inputs=[do_sample],
            outputs=[temperature, top_p, top_k],
            queue=False
        )
        
        def update_normalize_params_visibility(normalize_audio):
            return gr.update(visible=normalize_audio)
        
        normalize_audio.change(
            fn=update_normalize_params_visibility,
            inputs=[normalize_audio],
            outputs=[target_dB_FS],
            queue=False
        )
        
        def update_speaker_visibility(num_speakers):
            row_updates = []
            audio_updates = []
            for i in range(4):
                visible = (i < num_speakers)
                row_updates.append(gr.update(visible=visible))
                # Hide preview audio when speaker row is hidden
                audio_updates.append(gr.update(visible=False))
            return row_updates + audio_updates
        
        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_rows + speaker_preview_audios
        )
        
        # Preview voice functions
        def preview_voice_handler(speaker_name):
            """Handle voice preview."""
            if not speaker_name:
                return gr.update(value=None, visible=False)
            
            preview_result = demo_instance.preview_voice(speaker_name)
            if preview_result is None:
                return gr.update(value=None, visible=False)
            
            return gr.update(value=preview_result, visible=True)
        
        # Connect preview buttons
        for i in range(4):
            speaker_preview_buttons[i].click(
                fn=preview_voice_handler,
                inputs=[speaker_selections[i]],
                outputs=[speaker_preview_audios[i]],
                queue=False
            )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(model_name, num_speakers, script, *all_params):
            """Wrapper function to handle the streaming generation call."""
            try:
                # Extract parameters in order:
                # all_params = [speaker_1, speaker_2, speaker_3, speaker_4, 
                #               speech_rate, enable_bgm, bgm_volume, cfg_scale, ddpm_inference_steps,
                #               do_sample, temperature, top_p, top_k, refresh_negative, verbose,
                #               normalize_audio, target_dB_FS, seed]
                speakers = all_params[:4]
                speech_rate = all_params[4] if len(all_params) > 4 else GENERATION_CONFIG["speech_rate"]
                enable_bgm_val = all_params[5] if len(all_params) > 5 else BGM_CONFIG["enable_bgm"]
                bgm_volume_val = all_params[6] if len(all_params) > 6 else BGM_CONFIG["bgm_volume"]
                cfg_scale_val = all_params[7] if len(all_params) > 7 else GENERATION_CONFIG["cfg_scale"]
                ddpm_steps = all_params[8] if len(all_params) > 8 else GENERATION_CONFIG["ddpm_inference_steps"]
                do_sample_val = all_params[9] if len(all_params) > 9 else GENERATION_CONFIG["do_sample"]
                temperature_val = all_params[10] if len(all_params) > 10 else None
                top_p_val = all_params[11] if len(all_params) > 11 else None
                top_k_val = all_params[12] if len(all_params) > 12 else None
                refresh_negative_val = all_params[13] if len(all_params) > 13 else GENERATION_CONFIG["refresh_negative"]
                verbose_val = all_params[14] if len(all_params) > 14 else GENERATION_CONFIG["verbose"]
                normalize_audio_val = all_params[15] if len(all_params) > 15 else AUDIO_CONFIG["normalize_audio"]
                target_dB_FS_val = all_params[16] if len(all_params) > 16 else AUDIO_CONFIG["target_dB_FS"]
                seed_val = all_params[17] if len(all_params) > 17 else ADVANCED_CONFIG["seed"]
                
                # Clear outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), "🎙️ Starting generation...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # The generator will yield multiple times
                final_log = "Starting generation..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    model_name=model_name,
                    cfg_scale=cfg_scale_val,
                    speech_rate=speech_rate,
                    enable_bgm=enable_bgm_val,
                    bgm_volume=bgm_volume_val,
                    ddpm_inference_steps=int(ddpm_steps),
                    do_sample=do_sample_val,
                    temperature=temperature_val if do_sample_val else None,
                    top_p=top_p_val if do_sample_val else None,
                    top_k=int(top_k_val) if do_sample_val and top_k_val is not None else None,
                    refresh_negative=refresh_negative_val,
                    verbose=verbose_val,
                    normalize_audio=normalize_audio_val,
                    target_dB_FS=target_dB_FS_val,
                    seed=int(seed_val) if seed_val is not None and seed_val >= 0 else None
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

            except Exception as e:
                error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting new generation."""
            return None, gr.update(value=None, visible=False)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[model_selection, num_speakers, script_input] + speaker_selections + [
                speech_rate, enable_bgm, bgm_volume, cfg_scale, ddpm_inference_steps,
                do_sample, temperature, top_p, top_k, refresh_negative, verbose,
                normalize_audio, target_dB_FS, seed
            ],
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
            queue=True  # Enable Gradio's built-in queue
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=lambda: (None, None),
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )
        
        # Function to randomly select an example
        def load_random_example():
            """Randomly select and load an example script."""
            import random
            
            # Get available examples
            if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
                example_scripts = demo_instance.example_scripts
            else:
                # Fallback to default
                example_scripts = [
                    [2, "Speaker 1: Welcome to our AI podcast demonstration!\nSpeaker 2: Thanks for having me. This is exciting!"]
                ]
            
            # Randomly select one
            if example_scripts:
                selected = random.choice(example_scripts)
                num_speakers_value = selected[0]
                script_value = selected[1]
                
                # Return the values to update the UI
                return num_speakers_value, script_value
            
            # Default values if no examples
            return 2, ""
        
        # Connect random example button
        random_example_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[num_speakers, script_input],
            queue=False  # Don't queue this simple operation
        )
        
        # Add usage tips
        gr.Markdown("""
        ### 💡 **Usage Tips**
        
        - Click **🚀 Generate Podcast** to start audio generation
        - **Live Streaming** tab shows audio as it's generated (may have slight pauses)
        - **Complete Audio** tab provides the full, uninterrupted podcast after generation
        - During generation, you can click **🛑 Stop Generation** to interrupt the process
        - The streaming indicator shows real-time generation progress
        """)
        
        # Add example scripts
        gr.Markdown("### 📚 **Example Scripts**")
        
        # Use dynamically loaded examples if available, otherwise provide a default
        if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
            example_scripts = demo_instance.example_scripts
        else:
            # Fallback to a simple default example if no scripts loaded
            example_scripts = [
                [1, "Speaker 1: Welcome to our AI podcast demonstration! This is a sample script showing how VibeVoice can generate natural-sounding speech."]
            ]
        
        gr.Examples(
            examples=example_scripts,
            inputs=[num_speakers, script_input],
            label="Try these example scripts:"
        )

    return interface


# convert_to_16_bit_wav is now imported from utils module


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=f"Path to the VibeVoice model directory (default: {get_model_path()})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=f"Device for inference (default: {get_device()})",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=None,
        help=f"Number of inference steps for DDPM (default: {GENERATION_CONFIG['ddpm_inference_steps']})",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=GRADIO_CONFIG["share"],
        help="Share the demo publicly via Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=GRADIO_CONFIG["port"],
        help=f"Port to run the demo on (default: {GRADIO_CONFIG['port']})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file (not implemented yet)",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    # Validate config
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠️  Config validation warning: {e}")
        print("   Continuing with current config values...")
    
    # Set seed for reproducibility
    set_seed(ADVANCED_CONFIG["seed"])

    print("🎙️ Initializing VibeVoice Demo with Streaming Support...")
    print(f"📋 Using config from: demo/config.py")
    
    # Initialize demo instance (use args if provided, otherwise use config defaults)
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port {args.port}")
    print(f"📁 Model path: {demo_instance.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: {'ENABLED' if GRADIO_CONFIG['streaming_enabled'] else 'DISABLED'}")
    print(f"⚙️  Inference steps: {demo_instance.inference_steps}")
    print(f"📊 CFG Scale: {GENERATION_CONFIG['cfg_scale']} (range: {GENERATION_CONFIG['cfg_scale_min']}-{GENERATION_CONFIG['cfg_scale_max']})")
    print(f"🎵 Speech Rate: {GENERATION_CONFIG['speech_rate']}x (range: {GENERATION_CONFIG['speech_rate_min']}-{GENERATION_CONFIG['speech_rate_max']})")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=GRADIO_CONFIG["queue_max_size"],
            default_concurrency_limit=GRADIO_CONFIG["default_concurrency_limit"]
        ).launch(
            share=args.share,
            server_port=args.port,
            server_name=GRADIO_CONFIG["server_name"],
            show_error=GRADIO_CONFIG["show_error"],
            show_api=GRADIO_CONFIG["show_api"]
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()
