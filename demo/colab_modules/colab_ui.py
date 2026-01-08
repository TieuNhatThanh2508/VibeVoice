"""
UI Components for VibeVoice Colab Demo
ĐÂY LÀ NƠI BẠN CÓ THỂ CHỈNH SỬA GIAO DIỆN

Các phần chính:
1. create_demo_interface() - Tạo giao diện chính cho podcast generation
2. Các hàm helper để tạo UI components
3. CSS và styling có thể chỉnh sửa trong config
"""

import os
import shutil
import random
import gradio as gr
from typing import List

from .colab_generator import PodcastGenerator
from .colab_voice import VoiceManager
from .colab_config import config


def create_header_html() -> str:
    """
    Tạo HTML header cho giao diện
    CHỈNH SỬA ĐÂY để thay đổi header
    """
    return f"""
    <div style="text-align: center; margin: 20px auto; max-width: 800px;">
        <h1 style="font-size: 2.5em; margin-bottom: 10px;">{config.ui.header_title}</h1>
        <p style="font-size: 1.2em; color: #555; margin-bottom: 15px;">{config.ui.header_subtitle}</p>
        <a href="{config.ui.colab_link_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4285F4; color: white; border-radius: 6px; text-decoration: none; font-size: 1em;">{config.ui.colab_link_text}</a>
    </div>
    """


def create_settings_column(voice_manager: VoiceManager, audio_processor) -> tuple:
    """
    Tạo cột settings bên trái
    CHỈNH SỬA ĐÂY để thay đổi layout settings
    """
    with gr.Group():
        gr.Markdown(config.ui.podcast_settings_label)
        
        num_speakers = gr.Slider(
            minimum=config.ui.num_speakers_min,
            maximum=config.ui.num_speakers_max,
            value=config.ui.num_speakers_default,
            step=config.ui.num_speakers_step,
            label="Number of Speakers"
        )
        
        gr.Markdown(config.ui.speaker_selection_label)
        
        speaker_selections = []
        voice_previews = []  # Audio preview components
        available_voices = voice_manager.get_available_voices()
        defaults = config.ui.default_speakers
        
        for i in range(4):
            val = defaults[i] if i < len(defaults) and defaults[i] in available_voices else None
            
            # Create speaker dropdown
            speaker = gr.Dropdown(
                choices=available_voices,
                value=val,
                label=f"Speaker {i+1}",
                visible=(i < 2)
            )
            speaker_selections.append(speaker)
            
            # Add preview audio component right after each speaker dropdown
            preview_audio = gr.Audio(
                label=f"🎵 Preview Voice {i+1}",
                visible=(i < 2),
                interactive=False,
                type="filepath",  # Use filepath for direct file loading
                show_label=True,
                show_download_button=False,
                autoplay=False,
                elem_id=f"preview_audio_{i+1}"  # Add ID for debugging
            )
            voice_previews.append(preview_audio)
        
        # Upload custom voices
        with gr.Accordion("🎤 Upload Custom Voices", open=config.ui.upload_voices_accordion_open):
            upload_audio = gr.File(
                label="Upload Voice Samples",
                file_count="multiple",
                file_types=["audio"]
            )
            process_upload_btn = gr.Button(config.ui.upload_voices_btn)
        
        # Advanced settings
        with gr.Accordion("⚙️ Advanced Settings", open=config.ui.advanced_settings_accordion_open):
            cfg_scale = gr.Slider(
                minimum=config.model.cfg_scale_min,
                maximum=config.model.cfg_scale_max,
                value=config.model.default_cfg_scale,
                step=config.model.cfg_scale_step,
                label="CFG Scale"
            )
            speech_rate = gr.Slider(
                minimum=config.audio.speech_rate_min,
                maximum=config.audio.speech_rate_max,
                value=config.audio.default_speech_rate,
                step=config.audio.speech_rate_step,
                label=config.ui.speech_rate_label,
                info=f"1.0 = bình thường, >1.0 = nhanh hơn, <1.0 = chậm hơn (Mặc định: {config.audio.default_speech_rate})"
            )
            remove_silence_checkbox = gr.Checkbox(
                label="Trim Silence from Podcast",
                value=False
            )
    
    return num_speakers, speaker_selections, voice_previews, upload_audio, process_upload_btn, cfg_scale, speech_rate, remove_silence_checkbox


def create_generation_column() -> tuple:
    """
    Tạo cột generation bên phải
    CHỈNH SỬA ĐÂY để thay đổi layout generation
    """
    with gr.Group():
        gr.Markdown(config.ui.script_input_label)
        
        script_input = gr.Textbox(
            label="Conversation Script",
            placeholder=config.ui.script_placeholder,
            lines=config.ui.script_input_lines
        )
        
        with gr.Row():
            random_example_btn = gr.Button(
                config.ui.random_example_btn,
                scale=1
            )
            generate_btn = gr.Button(
                config.ui.generate_btn,
                variant="primary",
                scale=2
            )
        
        stop_btn = gr.Button(
            config.ui.stop_btn,
            variant="stop",
            visible=False
        )
        
        gr.Markdown(config.ui.generated_output_label)
        audio_output = gr.Audio(label="Play Generated Podcast")
        
        with gr.Accordion(config.ui.download_files_label, open=config.ui.download_files_accordion_open):
            download_file = gr.File(label="Download Audio File (.wav)")
            json_file_output = gr.File(label="Download Timestamps (.json)")
    
    return script_input, random_example_btn, generate_btn, stop_btn, audio_output, download_file, json_file_output


def create_usage_tips_section(generator: PodcastGenerator) -> gr.Examples:
    """
    Tạo section usage tips và examples
    CHỈNH SỬA ĐÂY để thay đổi tips và examples
    """
    with gr.Accordion(config.ui.usage_tips_label, open=config.ui.usage_tips_accordion_open):
        gr.Markdown("""- **Upload Your Own Voices:** Create your own podcast with custom voice samples.  
- **Timestamps:** Useful if you want to generate a video using Wan2.2 or other tools. The timestamps let you automatically separate each speaker (splitting the long podcast into smaller chunks), pass the audio clips to your video generation model, and then merge the generated video clips into a full podcast video (e.g., using FFmpeg + any video generation model such as image+audio → video).""")
        
        # Load example scripts
        example_scripts = []
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.file.text_examples_dir)
        if os.path.exists(examples_dir):
            txt_files = sorted([f for f in os.listdir(examples_dir) if f.lower().endswith('.txt')])
            for txt_file in txt_files:
                try:
                    with open(os.path.join(examples_dir, txt_file), 'r', encoding='utf-8') as f:
                        script = f.read().strip()
                    if script:
                        num_speakers = VoiceManager.get_num_speakers_from_script(script)
                        example_scripts.append([num_speakers, script])
                except Exception as e:
                    print(f"Error loading example {txt_file}: {e}")
        
        return gr.Examples(
            examples=example_scripts,
            inputs=[],  # Will be set in create_demo_interface
            label="Try these example scripts:"
        )


def create_demo_interface(generator: PodcastGenerator, voice_manager: VoiceManager, audio_processor):
    """
    Tạo giao diện chính cho Gradio
    ĐÂY LÀ HÀM CHÍNH ĐỂ TẠO GIAO DIỆN - CHỈNH SỬA ĐÂY ĐỂ THAY ĐỔI TOÀN BỘ UI
    
    Args:
        generator: PodcastGenerator instance
        voice_manager: VoiceManager instance
        audio_processor: AudioProcessor instance
    
    Returns:
        gr.Blocks: Gradio interface
    """
    with gr.Blocks(title=config.ui.app_title) as interface:
        # Header
        gr.HTML(create_header_html())
        
        # Main content
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1):
                num_speakers, speaker_selections, voice_previews, upload_audio, process_upload_btn, cfg_scale, speech_rate, remove_silence_checkbox = create_settings_column(voice_manager, audio_processor)
            
            # Right column - Generation
            with gr.Column(scale=2):
                script_input, random_example_btn, generate_btn, stop_btn, audio_output, download_file, json_file_output = create_generation_column()
        
        # Usage tips and examples
        examples = create_usage_tips_section(generator)
        
        # Event handlers
        def process_and_refresh_voices(uploaded_files):
            """Handle voice upload"""
            if not uploaded_files:
                return [gr.update() for _ in speaker_selections] + [gr.update() for _ in voice_previews] + [None]
            
            voices_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.file.voices_dir)
            for f in uploaded_files:
                shutil.copy(f.name, os.path.join(voices_dir, os.path.basename(f.name)))
            
            voice_manager.setup_voice_presets()
            new_choices = voice_manager.get_available_voices()
            return [gr.update(choices=new_choices) for _ in speaker_selections] + [gr.update() for _ in voice_previews] + [None]
        
        def update_speaker_visibility(num):
            """Update speaker visibility based on number"""
            num_int = int(num)
            dropdown_updates = []
            preview_updates = []
            
            for i in range(4):
                is_visible = (i < num_int)
                # Update dropdown visibility
                dropdown_updates.append(gr.update(visible=is_visible))
                # Update preview audio visibility
                preview_updates.append(gr.update(visible=is_visible))
            
            return dropdown_updates + preview_updates
        
        def preview_voice(voice_name):
            """Preview selected voice"""
            try:
                if not voice_name:
                    return None
                
                if voice_name not in voice_manager.available_voices:
                    print(f"Warning: Voice '{voice_name}' not found in available voices")
                    return None
                
                voice_path = voice_manager.get_voice_path(voice_name)
                if not voice_path:
                    print(f"Warning: No path found for voice '{voice_name}'")
                    return None
                
                # Normalize path
                voice_path = os.path.normpath(voice_path)
                
                # Convert to absolute path if relative
                if not os.path.isabs(voice_path):
                    # Get base directory (demo/)
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    voice_path = os.path.join(base_dir, voice_path)
                    voice_path = os.path.normpath(voice_path)
                
                if not os.path.exists(voice_path):
                    print(f"Warning: Voice file not found: {voice_path}")
                    return None
                
                # Verify it's a valid audio file
                if not os.path.isfile(voice_path):
                    print(f"Warning: Voice path is not a file: {voice_path}")
                    return None
                
                # Verify file is not empty
                if os.path.getsize(voice_path) == 0:
                    print(f"Warning: Voice file is empty: {voice_path}")
                    return None
                
                print(f"Preview voice: {voice_name} -> {voice_path}")
                # Return absolute path for Gradio
                return os.path.abspath(voice_path)
                
            except Exception as e:
                print(f"Error in preview_voice for '{voice_name}': {e}")
                import traceback
                traceback.print_exc()
                return None
        
        def load_random_example():
            """Load random example script"""
            examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.file.text_examples_dir)
            example_scripts = []
            if os.path.exists(examples_dir):
                txt_files = sorted([f for f in os.listdir(examples_dir) if f.lower().endswith('.txt')])
                for txt_file in txt_files:
                    try:
                        with open(os.path.join(examples_dir, txt_file), 'r', encoding='utf-8') as f:
                            script = f.read().strip()
                        if script:
                            num_speakers = VoiceManager.get_num_speakers_from_script(script)
                            example_scripts.append([num_speakers, script])
                    except Exception as e:
                        print(f"Error loading example {txt_file}: {e}")
            
            if example_scripts:
                return random.choice(example_scripts)
            return (2, "Speaker 0: No examples loaded.")
        
        # Combined function to update visibility and reload previews
        def update_speakers_and_previews(num, speaker1, speaker2, speaker3, speaker4):
            """Update speaker visibility and reload previews in one go"""
            num_int = int(num)
            speaker_values = [speaker1, speaker2, speaker3, speaker4]
            all_updates = []
            
            # First update dropdown visibility
            for i in range(4):
                is_visible = (i < num_int)
                all_updates.append(gr.update(visible=is_visible))
            
            # Then update preview visibility and values
            for i in range(4):
                is_visible = (i < num_int)
                if is_visible and speaker_values[i]:
                    # If visible and has value, load preview
                    try:
                        voice_path = preview_voice(speaker_values[i])
                        if voice_path:
                            all_updates.append(gr.update(value=voice_path, visible=True))
                        else:
                            # If preview failed, show empty but visible
                            all_updates.append(gr.update(value=None, visible=True))
                    except Exception as e:
                        print(f"Error loading preview for speaker {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Show empty preview instead of error
                        all_updates.append(gr.update(value=None, visible=True))
                else:
                    # Hide or clear preview
                    all_updates.append(gr.update(value=None, visible=is_visible))
            
            return all_updates
        
        # Connect events
        num_speakers.change(
            fn=update_speakers_and_previews,
            inputs=[num_speakers] + speaker_selections,
            outputs=speaker_selections + voice_previews
        )
        
        # Connect voice preview for each speaker dropdown
        for speaker_dropdown, preview_audio in zip(speaker_selections, voice_previews):
            # When dropdown changes, update preview with error handling
            def make_preview_handler():
                """Create a preview handler"""
                def handler(voice_name):
                    try:
                        if not voice_name:
                            return None
                        
                        result = preview_voice(voice_name)
                        # Always return a valid value (None is acceptable for Gradio)
                        return result
                    except Exception as e:
                        print(f"Error in preview handler for '{voice_name}': {e}")
                        import traceback
                        traceback.print_exc()
                        # Return None instead of raising error
                        return None
                return handler
            
            # Create handler for this specific dropdown
            handler = make_preview_handler()
            
            speaker_dropdown.change(
                fn=handler,
                inputs=speaker_dropdown,
                outputs=preview_audio
            )
        
        # Load initial previews on page load
        def load_initial_previews():
            """Load initial previews for visible speakers"""
            preview_updates = []
            for i in range(4):
                if i < 2 and speaker_selections[i].value:  # First 2 speakers are visible by default
                    try:
                        voice_path = preview_voice(speaker_selections[i].value)
                        if voice_path:
                            preview_updates.append(gr.update(value=voice_path))
                        else:
                            preview_updates.append(gr.update())
                    except Exception as e:
                        print(f"Error loading initial preview for speaker {i+1}: {e}")
                        preview_updates.append(gr.update())
                else:
                    preview_updates.append(gr.update())
            return preview_updates
        
        interface.load(
            fn=load_initial_previews,
            outputs=voice_previews,
            queue=False
        )
        
        process_upload_btn.click(
            fn=process_and_refresh_voices,
            inputs=upload_audio,
            outputs=speaker_selections + voice_previews + [upload_audio]
        )
        
        generate_btn.click(
            fn=generator.generate_podcast_with_timestamps,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale, speech_rate, remove_silence_checkbox],
            outputs=[audio_output, download_file, json_file_output, generate_btn, stop_btn],
        )
        
        stop_btn.click(fn=generator.stop_audio_generation)
        
        random_example_btn.click(
            fn=load_random_example,
            outputs=[num_speakers, script_input]
        )
        
        # Update examples inputs
        if examples:
            examples.inputs = [num_speakers, script_input]
    
    return interface

