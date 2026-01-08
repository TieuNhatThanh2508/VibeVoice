# Product Context: VibeVoice
*Version: 1.0*
*Updated: 2025-01-27*

## Problem Statement
Traditional Text-to-Speech (TTS) systems face significant challenges in:
- **Scalability**: Difficulty processing long-form content efficiently
- **Speaker Consistency**: Maintaining consistent voice characteristics across long conversations
- **Natural Turn-Taking**: Generating natural dialogue flow in multi-speaker scenarios
- **Computational Efficiency**: Processing long audio sequences without excessive computational cost

VibeVoice addresses these challenges by using continuous speech tokenizers at ultra-low frame rates and a next-token diffusion framework.

## User Personas

### Research Scientist
- **Demographics**: PhD-level researchers, ML engineers
- **Goals**: Experiment with long-form TTS, test multi-speaker scenarios, generate research demos
- **Pain Points**: Need for high-quality, long-form audio generation for research purposes

### Content Creator
- **Demographics**: Podcast creators, video producers, content developers
- **Goals**: Generate podcast-style content, create multi-speaker conversations, produce long-form audio
- **Pain Points**: Time-consuming manual voice recording, need for consistent voice characteristics

### Developer/Engineer
- **Demographics**: Software developers, AI engineers
- **Goals**: Integrate TTS into applications, build voice-enabled products, experiment with voice synthesis
- **Pain Points**: Limited by traditional TTS capabilities, need for scalable solutions

## User Experience Goals
- **Ease of Use**: Simple interface for generating podcasts from text scripts
- **Flexibility**: Support for 1-4 speakers with custom voice samples
- **Quality**: High-fidelity audio generation with natural prosody
- **Efficiency**: Fast generation with streaming support for real-time feedback
- **Reliability**: Consistent speaker voices across long conversations

## Key Features

### 1. Long-Form Generation
- Generate audio up to 90 minutes (1.5B model) or 45 minutes (7B model)
- Efficient processing through ultra-low frame rate tokenization (7.5 Hz)

### 2. Multi-Speaker Support
- Support for 1-4 distinct speakers in conversations
- Voice cloning from reference audio samples
- Speaker consistency across long conversations

### 3. Streaming Generation
- Real-time audio generation with AudioStreamer
- Progressive audio playback during generation
- Stop/start control for user interaction

### 4. Voice Cloning
- Reference audio samples for speaker embedding
- Voice presets system for easy selection
- Custom voice upload support

### 5. Cross-Lingual Support
- English language support
- Chinese language support
- Potential for additional languages

### 6. Multiple Interfaces
- **Gradio Web Interface**: Full-featured demo with streaming
- **Command-Line Tool**: Batch processing from text files
- **Google Colab**: Cloud-based experimentation with auto-save

## Success Metrics
- **Audio Quality**: High MOS (Mean Opinion Score) ratings
- **Generation Speed**: Real-time factor (RTF) < 1.0 for efficient generation
- **Speaker Consistency**: Maintained across 90-minute conversations
- **User Adoption**: Research community usage and feedback
- **Model Performance**: Successful generation of long-form, multi-speaker content

## Use Cases

### 1. Research Demonstrations
- Generate podcast-style demos for research presentations
- Test multi-speaker conversation scenarios
- Experiment with long-form audio generation

### 2. Content Creation
- Generate podcast scripts with multiple speakers
- Create audio content for videos or presentations
- Produce long-form conversational content

### 3. Voice Synthesis Applications
- Integrate into voice-enabled applications
- Build conversational AI systems
- Create personalized voice experiences

## Limitations and Risks

### Technical Limitations
- **Language Support**: Currently limited to English and Chinese
- **Non-Speech Audio**: Does not handle background noise, music, or sound effects
- **Overlapping Speech**: Does not explicitly model overlapping speech segments
- **Commercial Use**: Not recommended without further testing

### Ethical Considerations
- **Deepfakes Risk**: High-quality synthetic speech can be misused
- **Disinformation**: Potential for creating convincing fake audio content
- **Responsible Use**: Users must ensure transcripts are reliable and disclose AI-generated content

## Target Audience
- **Primary**: Research scientists and ML engineers
- **Secondary**: Content creators and developers
- **Tertiary**: General public interested in TTS technology

---

*This document explains why the project exists and what problems it solves.*

