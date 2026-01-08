# VibeVoice Memory Bank

This directory contains comprehensive documentation about the VibeVoice project, maintained automatically by the AI assistant to ensure continuity across coding sessions.

## File Structure

### Core Documentation
- **projectbrief.md**: Foundation document defining core requirements and goals
- **productContext.md**: Why this project exists and problems it solves
- **systemPatterns.md**: System architecture and key technical decisions
- **techContext.md**: Technologies used and development setup
- **activeContext.md**: Current work focus and next steps
- **progress.md**: What works, what's left to build, and known issues

### Implementation Plans
- **implementation-plans/**: Saved PLAN mode checklists (when created)

## Quick Reference

### Project Overview
VibeVoice is a novel framework for generating **expressive**, **long-form**, **multi-speaker** conversational audio (podcasts) from text. It uses continuous speech tokenizers at ultra-low frame rates (7.5 Hz) and a next-token diffusion framework.

### Key Capabilities
- Generate audio up to 90 minutes long
- Support 1-4 distinct speakers
- Maintain speaker consistency across long conversations
- Real-time streaming generation
- Voice cloning from reference samples

### Technology Stack
- **Base LLM**: Qwen2.5 (1.5B or 7B parameters)
- **Framework**: PyTorch, Transformers 4.51.3
- **Diffusion**: Next-token diffusion with DPM Solver
- **Tokenization**: Acoustic + Semantic tokenizers at 7.5 Hz

### Entry Points
- **Gradio Demo**: `demo/gradio_demo.py` - Full web interface
- **Colab Version**: `demo/colab.py` - Google Colab optimized
- **CLI Inference**: `demo/inference_from_file.py` - Batch processing

### Core Modules
- **Models**: `vibevoice/modular/modeling_vibevoice*.py`
- **Tokenizers**: `vibevoice/modular/modular_vibevoice_tokenizer*.py`
- **Processors**: `vibevoice/processor/vibevoice_processor.py`
- **Scheduling**: `vibevoice/schedule/dpm_solver.py`

## Usage

When starting a new session or task:
1. Read all memory bank files to understand project context
2. Check `activeContext.md` for current focus
3. Review `progress.md` for what's working and what needs work
4. Reference `systemPatterns.md` for architecture understanding

## Maintenance

Memory bank files are automatically updated when:
- New patterns are discovered
- Significant changes are implemented
- User requests with **update memory bank**
- Context needs clarification

---

*This memory bank ensures continuity and context preservation across coding sessions.*

