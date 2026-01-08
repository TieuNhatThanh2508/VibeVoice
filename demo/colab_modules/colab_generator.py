"""
Generation logic for VibeVoice Colab Demo
"""

import json
import os
import re
import traceback
from typing import Iterator, Tuple, Optional
import numpy as np
import soundfile as sf
import gradio as gr

from .colab_model import ModelManager
from .colab_voice import VoiceManager
from .colab_audio import AudioProcessor
from ..colab_utils import drive_save, generate_file_name
from .colab_config import config


class PodcastGenerator:
    """Handles podcast generation with timestamps"""
    
    def __init__(self, model_manager: ModelManager, voice_manager: VoiceManager, audio_processor: AudioProcessor):
        """
        Initialize generator
        
        Args:
            model_manager: ModelManager instance
            voice_manager: VoiceManager instance
            audio_processor: AudioProcessor instance
        """
        self.model_manager = model_manager
        self.voice_manager = voice_manager
        self.audio_processor = audio_processor
        
        self.is_generating = False
        self.stop_generation = False
    
    def format_script(self, script: str, num_speakers: int) -> list:
        """
        Format script with speaker labels
        
        Args:
            script: Raw script text
            num_speakers: Number of speakers
        
        Returns:
            list: Formatted script lines
        """
        lines = script.strip().split('\n')
        formatted_script_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if re.match(r'Speaker\s*\d+:', line, re.IGNORECASE):
                formatted_script_lines.append(line)
            else:
                speaker_id = len(formatted_script_lines) % num_speakers
                formatted_script_lines.append(f"Speaker {speaker_id+1}: {line}")
        
        return formatted_script_lines
    
    def generate_podcast_with_timestamps(
        self,
        num_speakers: int,
        script: str,
        speaker_1: str,
        speaker_2: str,
        speaker_3: str,
        speaker_4: str,
        cfg_scale: float,
        remove_silence: bool,
        progress=gr.Progress()
    ) -> Iterator[Tuple]:
        """
        Generate podcast with timestamps
        
        Yields:
            Tuple: (audio_path, download_file, json_file, generate_btn_update, stop_btn_update)
        """
        # Initial UI state
        yield None, None, None, gr.update(visible=False), gr.update(visible=True)
        
        final_audio_path, final_json_path = None, None
        
        try:
            self.stop_generation = False
            self.is_generating = True
            
            # Validation
            if not script.strip():
                raise gr.Error("Error: Please provide a script.")
            
            script = script.replace("'", "'")
            
            if not 1 <= num_speakers <= 4:
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.voice_manager.available_voices:
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
            # Load voice samples
            voice_samples = [
                self.audio_processor.read_audio(self.voice_manager.get_voice_path(name))
                for name in selected_speakers
            ]
            
            if any(len(vs) == 0 for vs in voice_samples):
                raise gr.Error("Error: Failed to load one or more audio files.")
            
            # Format script
            formatted_script_lines = self.format_script(script, num_speakers)
            
            if not formatted_script_lines:
                raise gr.Error("Error: Script is empty after formatting.")
            
            # Prepare output files
            timestamps = {}
            current_time = 0.0
            sample_rate = config.audio.sample_rate
            
            base_filename = generate_file_name(formatted_script_lines[0])
            final_audio_path = base_filename + ".wav"
            final_json_path = base_filename + ".json"
            
            # Generate audio for each line
            with sf.SoundFile(
                final_audio_path,
                'w',
                samplerate=sample_rate,
                channels=config.audio.channels,
                subtype=config.audio.audio_subtype
            ) as audio_file:
                for i, line in enumerate(formatted_script_lines):
                    if self.stop_generation:
                        print("\n🚫 Generation interrupted by user. Finalizing partial files...")
                        break
                    
                    progress(i / len(formatted_script_lines), desc=f"Generating line {i+1}/{len(formatted_script_lines)}")
                    
                    match = re.match(r'Speaker\s*(\d+):\s*(.*)', line, re.IGNORECASE)
                    if not match:
                        continue
                    
                    speaker_idx = int(match.group(1)) - 1
                    text_content = match.group(2).strip()
                    
                    if not (0 <= speaker_idx < len(voice_samples)):
                        continue
                    
                    # Generate audio for this line
                    inputs = self.model_manager.processor(
                        text=[line],
                        voice_samples=[voice_samples[speaker_idx]],
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    output_waveform = self.model_manager.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.model_manager.processor.tokenizer,
                        generation_config=self.model_manager.get_generation_config(),
                        verbose=config.model.verbose,
                        refresh_negative=config.model.refresh_negative
                    )
                    
                    audio_np = output_waveform.speech_outputs[0].cpu().float().numpy().squeeze()
                    
                    # Trim silence if requested
                    if remove_silence:
                        audio_np = self.audio_processor.trim_silence_from_numpy(audio_np, sample_rate)
                    
                    duration = len(audio_np) / sample_rate
                    audio_file.write((audio_np * 32767).astype(np.int16))
                    
                    timestamps[str(i + 1)] = {
                        "text": text_content,
                        "speaker_id": speaker_idx + 1,
                        "start": current_time,
                        "end": current_time + duration
                    }
                    current_time += duration
            
            if not timestamps:
                self.is_generating = False
                if os.path.exists(final_audio_path):
                    os.remove(final_audio_path)
                yield None, None, None, gr.update(visible=True), gr.update(visible=False)
                return
            
            # Save timestamps
            progress(1.0, desc="Saving generated files...")
            with open(final_json_path, "w") as f:
                json.dump(timestamps, f, indent=2)
            
            # Save to Google Drive if on Colab
            try:
                drive_save(final_audio_path)
                drive_save(final_json_path)
            except Exception as e:
                print(f"Error saving files to Google Drive: {e}")
            
            message = "Partial" if self.stop_generation else "Full"
            print(f"\n✨ {message} generation successful!\n🎵 Audio: {final_audio_path}\n📄 Timestamps: {final_json_path}\n")
            
            self.is_generating = False
            yield final_audio_path, final_audio_path, final_json_path, gr.update(visible=True), gr.update(visible=False)
        
        except Exception as e:
            self.is_generating = False
            print(f"❌ An unexpected error occurred: {str(e)}")
            traceback.print_exc()
            try:
                if final_audio_path and os.path.exists(final_audio_path):
                    os.remove(final_audio_path)
                if final_json_path and os.path.exists(final_json_path):
                    os.remove(final_json_path)
            except Exception as cleanup_e:
                print(f"Error during cleanup after exception: {cleanup_e}")
            yield None, None, None, gr.update(visible=True), gr.update(visible=False)
    
    def stop_audio_generation(self):
        """Stop the current audio generation process"""
        if self.is_generating:
            self.stop_generation = True
            print("🛑 Audio generation stop requested")

