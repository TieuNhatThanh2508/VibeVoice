# VibeVoice Codebase Index
*Version: 1.0*
*Updated: 2025-01-27*

## Overview
This document provides a comprehensive index of the VibeVoice codebase structure, key files, and their purposes.

## Directory Structure

```
VibeVoice_me/
├── demo/                          # Demo applications
│   ├── colab.py                   # Google Colab optimized demo
│   ├── gradio_demo.py             # Full Gradio web interface
│   ├── inference_from_file.py    # Command-line batch processing
│   ├── example/                   # Example video outputs
│   ├── text_examples/            # Example text scripts
│   └── voices/                   # Voice sample files (.wav)
├── vibevoice/                    # Main package
│   ├── __init__.py
│   ├── configs/                  # Model configurations
│   │   ├── qwen2.5_1.5b_64k.json
│   │   └── qwen2.5_7b_32k.json
│   ├── modular/                  # Core model components
│   │   ├── __init__.py
│   │   ├── configuration_vibevoice.py
│   │   ├── modeling_vibevoice.py
│   │   ├── modeling_vibevoice_inference.py
│   │   ├── modular_vibevoice_diffusion_head.py
│   │   ├── modular_vibevoice_text_tokenizer.py
│   │   ├── modular_vibevoice_tokenizer.py
│   │   └── streamer.py
│   ├── processor/                # Input/output processing
│   │   ├── __init__.py
│   │   ├── vibevoice_processor.py
│   │   └── vibevoice_tokenizer_processor.py
│   ├── schedule/                 # Diffusion scheduling
│   │   ├── __init__.py
│   │   ├── dpm_solver.py
│   │   └── timestep_sampler.py
│   └── scripts/                 # Utility scripts
│       ├── __init__.py
│       └── convert_nnscaler_checkpoint_to_transformers.py
├── memory-bank/                 # Project documentation
├── Figures/                    # Project images
├── report/                     # Technical report
├── README.md                    # Project README
├── pyproject.toml              # Package configuration
└── LICENSE                      # License file
```

## Key Files Reference

### Demo Applications

#### `demo/gradio_demo.py`
- **Purpose**: Full-featured Gradio web interface with streaming support
- **Key Features**:
  - Real-time audio streaming during generation
  - Multi-speaker selection (1-4 speakers)
  - Voice preset management
  - Example script loading
  - Stop/start generation control
- **Main Classes**: `VibeVoiceDemo`
- **Entry Point**: `main()` function

#### `demo/colab.py`
- **Purpose**: Google Colab optimized version
- **Key Features**:
  - Auto-save to Google Drive
  - Model downloading from HuggingFace
  - Two-tab interface (Podcasting + Prompt Builder)
  - Timestamp generation for video tools
- **Main Classes**: `VibeVoiceDemo`
- **Entry Point**: `main()` function with Click CLI

#### `demo/inference_from_file.py`
- **Purpose**: Command-line batch processing from text files
- **Key Features**:
  - Parse text files with speaker labels
  - Map speaker names to voice files
  - Batch generation with metrics (RTF, token counts)
  - Voice mapper for flexible speaker assignment
- **Main Classes**: `VoiceMapper`
- **Entry Point**: `main()` function

### Core Model Components

#### `vibevoice/modular/configuration_vibevoice.py`
- **Purpose**: Configuration system for VibeVoice models
- **Key Classes**:
  - `VibeVoiceAcousticTokenizerConfig`: Acoustic tokenizer configuration
  - `VibeVoiceSemanticTokenizerConfig`: Semantic tokenizer configuration
  - `VibeVoiceDiffusionHeadConfig`: Diffusion head configuration
  - `VibeVoiceConfig`: Main compositional config
- **Pattern**: Compositional config with sub-configs

#### `vibevoice/modular/modeling_vibevoice.py`
- **Purpose**: Main model architecture
- **Key Classes**:
  - `VibeVoicePreTrainedModel`: Base model class
  - `VibeVoiceModel`: Main model combining LLM and diffusion
  - `SpeechConnector`: Connects speech features to LLM
- **Components**:
  - Language model (Qwen2.5)
  - Acoustic tokenizer
  - Semantic tokenizer
  - Diffusion head

#### `vibevoice/modular/modeling_vibevoice_inference.py`
- **Purpose**: Inference wrapper with generation methods
- **Key Classes**:
  - `VibeVoiceForConditionalGenerationInference`: Inference model
- **Features**:
  - Generation with streaming support
  - CFG (Classifier-Free Guidance) scaling
  - DPM solver integration
  - Stop signal handling

#### `vibevoice/modular/modular_vibevoice_tokenizer.py`
- **Purpose**: Acoustic and semantic tokenizers
- **Key Classes**:
  - `VibeVoiceAcousticTokenizerModel`: Acoustic tokenization
  - `VibeVoiceSemanticTokenizerModel`: Semantic tokenization
  - `VibeVoiceTokenizerStreamingCache`: Streaming cache support
- **Features**: Encoder/decoder architecture for speech tokenization

#### `vibevoice/modular/modular_vibevoice_text_tokenizer.py`
- **Purpose**: Text tokenization using Qwen tokenizer
- **Key Classes**:
  - `VibeVoiceTextTokenizer`: Text tokenizer wrapper
  - `VibeVoiceTextTokenizerFast`: Fast tokenizer variant
- **Base**: Qwen2.5 tokenizer

#### `vibevoice/modular/modular_vibevoice_diffusion_head.py`
- **Purpose**: Diffusion head for audio generation
- **Key Classes**:
  - `VibeVoiceDiffusionHead`: Diffusion-based generation head
- **Features**: Next-token diffusion for high-fidelity audio

#### `vibevoice/modular/streamer.py`
- **Purpose**: Audio streaming support
- **Key Classes**: `AudioStreamer`
- **Features**: Real-time audio chunk streaming

### Processors

#### `vibevoice/processor/vibevoice_processor.py`
- **Purpose**: Main processor combining text and audio
- **Key Classes**:
  - `VibeVoiceProcessor`: Main processor class
- **Features**:
  - Text tokenization
  - Audio preprocessing
  - Voice sample processing
  - Audio normalization
  - System prompt handling

#### `vibevoice/processor/vibevoice_tokenizer_processor.py`
- **Purpose**: Audio normalization and preprocessing
- **Key Classes**:
  - `AudioNormalizer`: Audio normalization
  - `VibeVoiceTokenizerProcessor`: Audio processor
- **Features**: Sample rate conversion, normalization, decibel adjustment

### Scheduling

#### `vibevoice/schedule/dpm_solver.py`
- **Purpose**: DPM Solver for diffusion scheduling
- **Key Classes**: `DPMSolverMultistepScheduler`
- **Features**: SDE-based diffusion sampling

#### `vibevoice/schedule/timestep_sampler.py`
- **Purpose**: Timestep sampling for diffusion
- **Key Classes**: Timestep sampling utilities
- **Features**: Timestep management for diffusion process

## Data Flow

### Generation Pipeline
```
Text Input (with speaker labels)
    ↓
VibeVoiceProcessor
    ├─→ Text Tokenizer (Qwen) → Text Tokens
    └─→ Audio Processor → Voice Embeddings
    ↓
VibeVoiceModel
    ├─→ LLM (Qwen2.5) → Hidden States
    ├─→ Diffusion Head → Acoustic Tokens
    └─→ Acoustic Tokenizer → Audio Waveform
    ↓
Audio Output (24kHz, mono)
```

### Streaming Flow
```
Generation Thread
    ↓
AudioStreamer
    ↓
Audio Chunks (real-time)
    ↓
Gradio Interface
```

## Configuration Files

### `vibevoice/configs/qwen2.5_1.5b_64k.json`
- Model: 1.5B parameters
- Context: 64K tokens
- Max Generation: ~90 minutes

### `vibevoice/configs/qwen2.5_7b_32k.json`
- Model: 7B parameters
- Context: 32K tokens
- Max Generation: ~45 minutes

## Common Tasks

### Running Gradio Demo
```bash
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share
```

### Running Colab Version
```bash
python demo/colab.py --model_path microsoft/VibeVoice-1.5B --share
```

### Batch Inference
```bash
python demo/inference_from_file.py \
    --model_path microsoft/VibeVoice-1.5B \
    --txt_path demo/text_examples/1p_abs.txt \
    --speaker_names Alice
```

## Key Patterns

### 1. Speaker Format
Text scripts use format: `Speaker 1: text content`

### 2. Voice Mapping
Voice files in `demo/voices/` are mapped by filename (without extension)

### 3. Streaming
AudioStreamer provides real-time chunks during generation

### 4. Configuration
Compositional config system with sub-configs for each component

### 5. Multi-Speaker
Sequential generation with speaker consistency maintained through voice embeddings

## Dependencies

### Core
- `torch`: PyTorch framework
- `transformers==4.51.3`: Model loading (version locked)
- `diffusers`: Diffusion schedulers
- `accelerate==1.6.0`: Distributed training

### Audio
- `librosa`: Audio processing
- `soundfile`: Audio I/O
- `scipy`: Scientific computing
- `numpy`: Numerical operations

### UI
- `gradio`: Web interface
- `av`: Audio/Video processing
- `aiortc`: Real-time communication

### Utilities
- `tqdm`: Progress bars
- `ml-collections`: Configuration
- `absl-py`: Utilities

---

*This index provides a comprehensive reference for navigating the VibeVoice codebase.*

