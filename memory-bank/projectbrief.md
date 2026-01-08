# Project Brief: VibeVoice
*Version: 1.0*
*Created: 2025-01-27*

## Project Overview
VibeVoice is a novel framework designed for generating **expressive**, **long-form**, **multi-speaker** conversational audio (podcasts) from text. It addresses challenges in traditional Text-to-Speech (TTS) systems, particularly in scalability, speaker consistency, and natural turn-taking.

## Core Requirements
- Generate long-form conversational audio (up to 90 minutes)
- Support multi-speaker conversations (up to 4 distinct speakers)
- Maintain speaker consistency across long conversations
- Use continuous speech tokenizers at ultra-low frame rate (7.5 Hz)
- Leverage next-token diffusion framework with LLM for context understanding
- Support both English and Chinese languages

## Success Criteria
- Generate high-quality speech up to 90 minutes long
- Support up to 4 distinct speakers in conversations
- Maintain natural turn-taking and dialogue flow
- Efficient processing of long sequences through tokenization

## Scope
### In Scope
- Text-to-speech generation for podcasts
- Multi-speaker conversation synthesis
- Long-form audio generation (45-90 minutes)
- Cross-lingual support (English, Chinese)
- Voice cloning from reference audio samples

### Out of Scope
- Background noise or music generation
- Overlapping speech segments
- Commercial deployment without further testing
- Languages other than English and Chinese

## Timeline
- Project Status: Active development
- Models Available: VibeVoice-1.5B (64K context, ~90 min), VibeVoice-7B (32K context, ~45 min)

## Stakeholders
- Microsoft: Primary developer
- Research Community: Target users

---

*This document serves as the foundation for the project and informs all other memory files.*

