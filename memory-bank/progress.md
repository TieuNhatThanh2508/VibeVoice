# Progress Tracker: VibeVoice
*Version: 1.0*
*Updated: 2025-01-27*

## Project Status
Overall Completion: Research/Development Phase

## What Works
- **Model Architecture**: Hybrid AR + diffusion architecture with Qwen2.5 LLM base
- **Multi-Speaker Generation**: Supports 1-4 distinct speakers in conversations
- **Long-Form Audio**: Can generate up to 90 minutes of audio (1.5B model) or 45 minutes (7B model)
- **Voice Cloning**: Reference audio samples used for speaker embedding
- **Streaming Generation**: Real-time audio generation with AudioStreamer
- **Gradio Demo**: Full-featured web interface with streaming support
- **File Inference**: Command-line tool for batch processing from text files
- **Colab Integration**: Google Colab optimized version with auto-save to Drive
- **Cross-Lingual Support**: English and Chinese language support
- **DPM Solver**: SDE-based diffusion scheduler for stable generation

## What's In Progress
- **Codebase Indexing**: Currently documenting structure and patterns
- **Documentation**: Building comprehensive memory bank
- **Model Variants**: VibeVoice-0.5B-Streaming model in development

## What's Left To Build
- **Commercial Deployment**: Requires further testing and development
- **Overlapping Speech**: Not currently supported
- **Background Music/Noise**: Not handled by current model
- **Additional Languages**: Currently limited to English and Chinese
- **Real-Time Streaming API**: Could be enhanced for production use

## Known Issues
- **Transformers Version Lock**: Must use transformers==4.51.3 (later versions incompatible)
- **GPU Requirements**: Large models require significant VRAM
- **Flash Attention**: Optional but recommended for faster inference (T4 GPU limitations)
- **Language Limitations**: Only English and Chinese fully supported
- **Non-Speech Audio**: Model focuses solely on speech synthesis

## Milestones
- **Model Release**: VibeVoice-1.5B and VibeVoice-7B available on HuggingFace
- **Demo Launch**: Live playground available at https://aka.ms/VibeVoice-Demo
- **Technical Report**: Published in report/TechnicalReport.pdf

## Codebase Structure

### Core Modules
- `vibevoice/modular/`: Core model components
  - `modeling_vibevoice.py`: Main model architecture
  - `modeling_vibevoice_inference.py`: Inference wrapper with generation
  - `configuration_vibevoice.py`: Configuration system
  - `modular_vibevoice_tokenizer.py`: Acoustic and semantic tokenizers
  - `modular_vibevoice_text_tokenizer.py`: Text tokenization (Qwen-based)
  - `modular_vibevoice_diffusion_head.py`: Diffusion head for audio generation
  - `streamer.py`: Audio streaming support

### Processors
- `vibevoice/processor/`: Input/output processing
  - `vibevoice_processor.py`: Main processor combining text and audio
  - `vibevoice_tokenizer_processor.py`: Audio normalization and preprocessing

### Scheduling
- `vibevoice/schedule/`: Diffusion scheduling
  - `dpm_solver.py`: DPM Solver scheduler
  - `timestep_sampler.py`: Timestep sampling

### Demo Applications
- `demo/gradio_demo.py`: Full Gradio interface with streaming
- `demo/colab.py`: Google Colab optimized version
- `demo/inference_from_file.py`: Command-line batch processing

### Configuration
- `vibevoice/configs/`: Model configurations
  - `qwen2.5_1.5b_64k.json`: 1.5B model config (64K context)
  - `qwen2.5_7b_32k.json`: 7B model config (32K context)

---

*This document tracks what works, what's in progress, and what's left to build.*

