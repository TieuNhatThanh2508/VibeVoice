# Active Context: VibeVoice
*Version: 1.0*
*Updated: 2025-01-27*
*Current RIPER Mode: RESEARCH*

## Current Focus
Indexing and understanding the VibeVoice codebase structure. This is a Microsoft research project for generating long-form, multi-speaker conversational audio (podcasts) from text using a hybrid AR + diffusion architecture.

## Recent Changes
- Memory bank initialized with project structure
- Codebase indexed with key components identified
- Demo applications documented (gradio_demo.py, colab.py, inference_from_file.py)

## Active Decisions
- Using Qwen2.5 as base LLM for language understanding
- Ultra-low frame rate tokenization (7.5 Hz) for efficiency
- Next-token diffusion framework for high-fidelity audio generation
- Support for 1-4 speakers in conversations

## Next Steps
1. Complete codebase indexing
2. Document key implementation patterns
3. Identify entry points for common tasks
4. Map data flow through the system

## Current Challenges
- Understanding the complete tokenization pipeline
- Mapping the relationship between acoustic/semantic tokenizers
- Understanding streaming generation implementation

## Implementation Progress
- [✓] Read README and project structure
- [✓] Identified core components (tokenizers, models, processors)
- [✓] Documented demo applications
- [✓] Mapped configuration system
- [ ] Deep dive into tokenization pipeline
- [ ] Understand diffusion head implementation
- [ ] Document streaming mechanism

---

*This document captures the current state of work and immediate next steps.*

