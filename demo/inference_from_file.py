import argparse
import os
import re
from typing import List, Tuple, Union, Dict, Any
import time
import torch
import numpy as np
import librosa
import librosa.effects

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

# Import configuration
from .config import (
    MODEL_CONFIG,
    GENERATION_CONFIG,
    AUDIO_CONFIG,
    INFERENCE_CONFIG,
    ADVANCED_CONFIG,
    get_model_path,
    get_device,
    get_torch_dtype,
    validate_config,
)

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice file paths"""
    
    def __init__(self):
        self.setup_voice_presets()

        # change name according to our preset wav file
        new_dict = {}
        for name, path in self.voice_presets.items():
            
            if '_' in name:
                name = name.split('_')[0]
            
            if '-' in name:
                name = name.split('-')[-1]

            new_dict[name] = path
        self.voice_presets.update(new_dict)
        # print(list(self.voice_presets.keys()))

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
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        
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
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        # First try exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        
        # Default to first voice if no match found
        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse txt script content and extract speakers and their text
    Fixed pattern: Speaker 1, Speaker 2, Speaker 3, Speaker 4
    Returns: (scripts, speaker_numbers)
    """
    lines = txt_content.strip().split('\n')
    scripts = []
    speaker_numbers = []
    
    # Pattern to match "Speaker X:" format where X is a number
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    
    current_speaker = None
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            # If we have accumulated text from previous speaker, save it
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            
            # Start new speaker
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            # Continue text for current speaker
            if current_text:
                current_text += " " + line
            else:
                current_text = line
    
    # Don't forget the last speaker
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)
    
    return scripts, speaker_numbers


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Processor TXT Input Test")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=f"Path to the HuggingFace model directory (default: {get_model_path()})",
    )
    
    parser.add_argument(
        "--txt_path",
        type=str,
        default=INFERENCE_CONFIG["default_txt_path"],
        help=f"Path to the txt file containing the script (default: {INFERENCE_CONFIG['default_txt_path']})",
    )
    parser.add_argument(
        "--speaker_names",
        type=str,
        nargs='+',
        default=INFERENCE_CONFIG["default_speaker_names"],
        help=f"Speaker names in order (default: {INFERENCE_CONFIG['default_speaker_names']})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=INFERENCE_CONFIG["default_output_dir"],
        help=f"Directory to save output audio files (default: {INFERENCE_CONFIG['default_output_dir']})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=f"Device for inference (default: {get_device()})",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help=f"CFG (Classifier-Free Guidance) scale for generation (default: {GENERATION_CONFIG['cfg_scale']})",
    )
    parser.add_argument(
        "--speech_rate",
        type=float,
        default=None,
        help=f"Speech rate multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed) (default: {GENERATION_CONFIG['speech_rate']})",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=None,
        help=f"Number of DDPM inference steps (default: {GENERATION_CONFIG['ddpm_inference_steps']})",
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate config
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠️  Config validation warning: {e}")
        print("   Continuing with current config values...")

    # Initialize voice mapper
    voice_mapper = VoiceMapper()
    
    # Check if txt file exists
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return
    
    # Read and parse txt file
    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    
    # Parse the txt content to get speaker numbers
    scripts, speaker_numbers = parse_txt_script(txt_content)
    
    if not scripts:
        print("Error: No valid speaker scripts found in the txt file")
        return
    
    print(f"Found {len(scripts)} speaker segments:")
    for i, (script, speaker_num) in enumerate(zip(scripts, speaker_numbers)):
        print(f"  {i+1}. Speaker {speaker_num}")
        print(f"     Text preview: {script[:100]}...")
    
    # Map speaker numbers to provided speaker names
    speaker_name_mapping = {}
    speaker_names_list = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
    for i, name in enumerate(speaker_names_list, 1):
        speaker_name_mapping[str(i)] = name
    
    print(f"\nSpeaker mapping:")
    for speaker_num in set(speaker_numbers):
        mapped_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        print(f"  Speaker {speaker_num} -> {mapped_name}")
    
    # Map speakers to voice files using the provided speaker names
    voice_samples = []
    actual_speakers = []
    
    # Get unique speaker numbers in order of first appearance
    unique_speaker_numbers = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)
    
    for speaker_num in unique_speaker_numbers:
        speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        voice_path = voice_mapper.get_voice_path(speaker_name)
        voice_samples.append(voice_path)
        actual_speakers.append(speaker_name)
        print(f"Speaker {speaker_num} ('{speaker_name}') -> Voice: {os.path.basename(voice_path)}")
    
    # Prepare data for model
    full_script = '\n'.join(scripts)
    
    # Use config defaults if not provided via args
    model_path = args.model_path or get_model_path()
    device = args.device or get_device()
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else GENERATION_CONFIG["cfg_scale"]
    speech_rate = args.speech_rate if args.speech_rate is not None else GENERATION_CONFIG["speech_rate"]
    inference_steps = args.inference_steps if args.inference_steps is not None else GENERATION_CONFIG["ddpm_inference_steps"]
    
    # Load processor
    print(f"Loading processor & model from {model_path}")
    processor = VibeVoiceProcessor.from_pretrained(model_path)

    # Load model with config settings
    torch_dtype = get_torch_dtype()
    attn_impl = MODEL_CONFIG["attn_implementation"] if MODEL_CONFIG["attn_implementation"] else None
    
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=MODEL_CONFIG["device_map"],
        attn_implementation=attn_impl
    )

    model.eval()
    
    # Configure noise scheduler
    model.model.noise_scheduler = model.model.noise_scheduler.from_config(
        model.model.noise_scheduler.config,
        algorithm_type=MODEL_CONFIG["algorithm_type"],
        beta_schedule=MODEL_CONFIG["beta_schedule"]
    )
    model.set_ddpm_inference_steps(num_steps=inference_steps)

    if hasattr(model.model, 'language_model'):
       print(f"Language model attention: {model.model.language_model.config._attn_implementation}")
       
    # Prepare inputs for the model
    inputs = processor(
        text=[full_script],  # Wrap in list for batch processing
        voice_samples=[voice_samples],  # Wrap in list for batch processing
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    print(f"Starting generation with cfg_scale: {cfg_scale}")
    print(f"Using inference steps: {inference_steps}")
    print(f"Speech rate: {speech_rate}x")

    # Generate audio
    start_time = time.time()
    
    gen_config = {
        'do_sample': GENERATION_CONFIG["do_sample"],
    }
    if GENERATION_CONFIG["temperature"] is not None:
        gen_config['temperature'] = GENERATION_CONFIG["temperature"]
    if GENERATION_CONFIG["top_p"] is not None:
        gen_config['top_p'] = GENERATION_CONFIG["top_p"]
    if GENERATION_CONFIG["top_k"] is not None:
        gen_config['top_k'] = GENERATION_CONFIG["top_k"]
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config=gen_config,
        verbose=GENERATION_CONFIG["verbose"],
        refresh_negative=GENERATION_CONFIG["refresh_negative"],
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")
    
    # Calculate audio duration and additional metrics
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        sample_rate = AUDIO_CONFIG["sample_rate"]
        audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        
        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    else:
        print("No audio output generated")
    
    # Calculate token metrics
    input_tokens = inputs['input_ids'].shape[1]  # Number of input tokens
    output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
    generated_tokens = output_tokens - input_tokens
    
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")

    # Apply speech rate adjustment if needed
    audio_output = outputs.speech_outputs[0]
    sample_rate = AUDIO_CONFIG["sample_rate"]
    
    if speech_rate != 1.0 and AUDIO_CONFIG["enable_speech_rate"]:
        print(f"Adjusting speech rate to {speech_rate:.2f}x...")
        # Convert to numpy if tensor
        if torch.is_tensor(audio_output):
            audio_np = audio_output.float().cpu().numpy()
        else:
            audio_np = np.array(audio_output)
        
        # Ensure 1D
        if len(audio_np.shape) > 1:
            audio_np = audio_np.squeeze()
        
        # Apply time-stretching
        try:
            audio_np = librosa.effects.time_stretch(audio_np, rate=speech_rate)
            # Convert back to tensor if needed
            if torch.is_tensor(audio_output):
                audio_output = torch.from_numpy(audio_np).float()
            else:
                audio_output = audio_np
            print(f"Speech rate adjusted successfully")
        except Exception as e:
            print(f"Warning: Failed to adjust speech rate: {e}. Using original audio.")
    
    # Save output
    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    output_path = os.path.join(args.output_dir, f"{txt_filename}_generated.wav")
    if INFERENCE_CONFIG["auto_create_output_dir"]:
        os.makedirs(args.output_dir, exist_ok=True)
    
    processor.save_audio(
        audio_output,  # First (and only) batch item
        output_path=output_path,
    )
    print(f"Saved output to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {output_path}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"Speaker names: {args.speaker_names}")
    print(f"Number of unique speakers: {len(set(speaker_numbers))}")
    print(f"Number of segments: {len(scripts)}")
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"RTF (Real Time Factor): {rtf:.2f}x")
    print(f"Inference steps: {inference_steps}")
    print(f"CFG scale: {cfg_scale}")
    print(f"Speech rate: {speech_rate:.2f}x")
    print("="*50)
    
    print("="*50)

if __name__ == "__main__":
    main()
