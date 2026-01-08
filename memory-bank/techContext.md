# Technical Context: VibeVoice
*Version: 1.0*
*Updated: 2025-01-27*

## Technology Stack

### Core Framework
- **Python**: >=3.8
- **PyTorch**: Deep learning framework
- **Transformers**: 4.51.3 (specific version required)
- **Diffusers**: Diffusion model support
- **Accelerate**: 1.6.0 for distributed training

### Audio Processing
- **librosa**: Audio analysis and resampling
- **soundfile**: Audio I/O
- **scipy**: Scientific computing
- **numpy**: Numerical operations

### UI/Interface
- **Gradio**: Web interface for demos
- **av**: Audio/Video processing
- **aiortc**: Real-time communication (for streaming)

### Utilities
- **tqdm**: Progress bars
- **ml-collections**: Configuration management
- **absl-py**: Abseil Python utilities

## Development Environment Setup

### Recommended Setup
1. Use NVIDIA Deep Learning Container (PyTorch 24.07+)
2. Install flash-attention if not included in container
3. Clone repository and install with `pip install -e .`

### Docker Environment
```bash
sudo docker run --privileged --net=host --ipc=host \
  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \
  --gpus all --rm -it nvcr.io/nvidia/pytorch:24.07-py3
```

## Dependencies

### Critical Dependencies
- **torch**: Core deep learning
- **transformers==4.51.3**: Model loading and tokenization (version locked)
- **accelerate==1.6.0**: Distributed training support
- **diffusers**: Diffusion schedulers
- **librosa**: Audio processing
- **gradio**: Demo interface

### Optional Dependencies
- **flash-attn**: Flash attention for faster inference (requires compatible GPU)
- **pydub**: Audio manipulation (used in colab.py)

## Technical Constraints

1. **Transformers Version**: Must use 4.51.3 (later versions may be incompatible)
2. **GPU Requirements**: CUDA-capable GPU recommended (T4+ for best performance)
3. **Memory**: Large models (1.5B, 7B) require significant VRAM
4. **Flash Attention**: Optional but recommended for faster inference
5. **Audio Format**: 24kHz sample rate, mono channel

## Build and Deployment

### Installation
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/
pip install -e .
```

### Model Loading
- Models available on HuggingFace:
  - `microsoft/VibeVoice-1.5B` (64K context, ~90 min)
  - `WestZhang/VibeVoice-Large-pt` (7B, 32K context, ~45 min)

### Inference Configuration
- Default inference steps: 5-10
- CFG scale: 1.3 (default)
- Device: CUDA (with CPU fallback)

## Testing Approach

### Demo Applications
- **Gradio Demo**: Interactive web interface
- **File Inference**: Command-line batch processing
- **Colab Notebook**: Cloud-based experimentation

### Example Scripts
- Located in `demo/text_examples/`
- Support 1-4 speakers
- Various conversation lengths

---

*This document describes the technologies used in the project and how they're configured.*

