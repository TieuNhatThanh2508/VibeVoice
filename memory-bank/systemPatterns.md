# System Patterns: VibeVoice
*Version: 1.0*
*Updated: 2025-01-27*

## Architecture Overview
VibeVoice uses a hybrid architecture combining:
1. **Continuous Speech Tokenizers** (Acoustic & Semantic) - Operating at 7.5 Hz frame rate
2. **Large Language Model (LLM)** - Qwen2.5 based decoder for understanding textual context
3. **Diffusion Head** - Generates high-fidelity acoustic details using next-token diffusion
4. **DPM Solver** - SDE-based scheduler for diffusion process

## Key Components

### 1. Tokenizers (`vibevoice/modular/`)
- **Acoustic Tokenizer** (`modular_vibevoice_tokenizer.py`): Encodes/decodes acoustic features
- **Semantic Tokenizer** (`modular_vibevoice_tokenizer.py`): Encodes/decodes semantic information
- **Text Tokenizer** (`modular_vibevoice_text_tokenizer.py`): Processes text input using Qwen tokenizer

### 2. Core Models (`vibevoice/modular/`)
- **VibeVoiceModel** (`modeling_vibevoice.py`): Main model combining LLM and diffusion head
- **VibeVoiceForConditionalGenerationInference** (`modeling_vibevoice_inference.py`): Inference wrapper with generation methods
- **VibeVoiceDiffusionHead** (`modular_vibevoice_diffusion_head.py`): Diffusion-based audio generation head

### 3. Processors (`vibevoice/processor/`)
- **VibeVoiceProcessor** (`vibevoice_processor.py`): Main processor combining text and audio processing
- **VibeVoiceTokenizerProcessor** (`vibevoice_tokenizer_processor.py`): Audio normalization and preprocessing

### 4. Scheduling (`vibevoice/schedule/`)
- **DPMSolverMultistepScheduler** (`dpm_solver.py`): SDE-based diffusion scheduler
- **TimestepSampler** (`timestep_sampler.py`): Manages diffusion timesteps

### 5. Demo Applications (`demo/`)
- **gradio_demo.py**: Full-featured Gradio interface with streaming support
- **colab.py**: Google Colab optimized version with auto-save to Drive
- **inference_from_file.py**: Command-line inference from text files

## Design Patterns in Use

### 1. Modular Architecture
- Separate tokenizers for acoustic, semantic, and text processing
- Compositional config system with sub-configs
- Clear separation between training and inference models

### 2. Next-Token Diffusion
- LLM generates next token predictions
- Diffusion head refines acoustic details
- Streaming support for real-time generation

### 3. Voice Cloning
- Reference audio samples used for speaker embedding
- Voice presets system for easy speaker selection
- Support for custom voice uploads

### 4. Multi-Speaker Handling
- Speaker labels in text format: "Speaker 1:", "Speaker 2:", etc.
- Voice mapping system to assign voices to speakers
- Sequential generation with speaker consistency

## Data Flow

```
Text Input → Text Tokenizer → LLM (Qwen2.5) → Diffusion Head → Acoustic Tokenizer → Audio Output
                ↓
         Voice Samples → Audio Processor → Speaker Embeddings
```

## Key Technical Decisions

1. **Ultra-Low Frame Rate (7.5 Hz)**: Enables efficient processing of long sequences
2. **Qwen2.5 as Base LLM**: Provides strong language understanding
3. **SDE-DPMSolver++**: Fast and stable diffusion sampling
4. **Compositional Config**: Allows flexible model configuration
5. **Streaming Support**: Real-time audio generation capability

## Component Relationships

- `VibeVoiceConfig` → Composes sub-configs (acoustic, semantic, decoder, diffusion)
- `VibeVoiceModel` → Contains LLM decoder + Diffusion Head + Tokenizers
- `VibeVoiceProcessor` → Wraps text tokenizer + audio processor
- `VibeVoiceForConditionalGenerationInference` → Extends model with generation methods

---

*This document captures the system architecture and design patterns used in the project.*

