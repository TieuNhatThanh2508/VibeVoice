"""
Prompt Builder UI for VibeVoice Colab Demo
ĐÂY LÀ NƠI BẠN CÓ THỂ CHỈNH SỬA GIAO DIỆN PROMPT BUILDER
"""

import gradio as gr
from .colab_config import config


def build_conversation_prompt(topic, *speaker_names):
    """
    Generate prompt for LLM to create podcast script
    CHỈNH SỬA ĐÂY để thay đổi format prompt
    """
    names = [name for name in speaker_names if name and name.strip()]

    # Error checking
    if not topic or not topic.strip():
        return "Error: Please provide a topic."
    if not names:
        return "Error: Please provide at least one speaker name."

    num_speakers = len(names)
    speaker_mapping_str = "Speaker mapping (for context only, DO NOT use these names as labels):\n"
    for i, name in enumerate(names):
        speaker_mapping_str += f"- Speaker {i+1} = {name}\n"
    
    speaker_labels = [f"\"Speaker {i+1}:\"" for i in range(num_speakers)]
    
    introductions_str = ""
    for i, name in enumerate(names):
        introductions_str += f"  - Speaker {i+1} introduces themselves by saying: \"I'm {name}...\"\n"
        
    example_str = "STRICT Example (follow this format exactly):\n"
    example_str += f"Speaker 1: Hi everyone, I'm {names[0]}, and I'm excited to be here today.\n"
    if num_speakers > 1:
        for i in range(1, num_speakers):
            example_str += f"Speaker {i+1}: And I'm {names[i]}. Thanks for joining us.\n"
    example_str += "Speaker 1: So, let's dive into our topic...\n"
    
    prompt = f"""
You are a professional podcast scriptwriter. 
Write a natural, engaging conversation between {num_speakers} speakers on the topic: "{topic}".
{speaker_mapping_str}
Formatting Rules:
- You MUST always format dialogue with {', '.join(speaker_labels)} ONLY. 
- Never replace the labels with real names. The labels stay exactly as they are.
- At the beginning:
{introductions_str}
- During the conversation, they may occasionally mention each other's names ({', '.join(names)}) naturally in the dialogue, but the labels must remain unchanged.
- Do not add narration, descriptions, or any extra formatting.
{example_str}
"""
    return prompt


def update_speaker_name_visibility(num_speakers):
    """
    Show or hide speaker name textboxes based on slider value
    CHỈNH SỬA ĐÂY để thay đổi behavior
    """
    num = int(num_speakers)
    updates = []
    for i in range(4):
        if i < num:
            updates.append(gr.update(visible=True))
        else:
            updates.append(gr.update(visible=False, value=""))
    
    return tuple(updates)


def create_prompt_builder_ui():
    """
    Tạo giao diện Prompt Builder
    ĐÂY LÀ HÀM CHÍNH ĐỂ TẠO PROMPT BUILDER UI - CHỈNH SỬA ĐÂY ĐỂ THAY ĐỔI UI
    """
    with gr.Blocks(title="Prompt Builder") as demo:
        # Header
        gr.HTML(f"""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">{config.prompt_builder.title}</h1>
            <p style="font-size: 1.2em; color: #555;">{config.prompt_builder.subtitle}</p>
        </div>""")
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                topic = gr.Textbox(
                    label="Topic",
                    placeholder=config.prompt_builder.topic_placeholder
                )
                
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Number of Speakers"
                )
                
                with gr.Group():
                    speaker_textboxes = [
                        gr.Textbox(
                            label=f"Speaker {i+1} Name",
                            visible=(i < 2),
                            placeholder=config.prompt_builder.speaker_name_placeholder.format(i=i+1)
                        )
                        for i in range(4)
                    ]
                
                gen_btn = gr.Button(
                    config.ui.generate_prompt_btn,
                    variant="primary"
                )
                
                # Examples
                gr.Examples(
                    examples=config.prompt_builder.example_prompts,
                    inputs=[topic, num_speakers] + speaker_textboxes,
                    label="Quick Examples"
                )
            
            # Right column - Output
            with gr.Column(scale=2):
                output_prompt = gr.Textbox(
                    label="Generated Prompt",
                    lines=config.ui.prompt_output_lines,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Event handlers
        num_speakers.change(
            fn=update_speaker_name_visibility,
            inputs=num_speakers,
            outputs=speaker_textboxes
        )
        
        gen_btn.click(
            fn=build_conversation_prompt,
            inputs=[topic] + speaker_textboxes,
            outputs=[output_prompt]
        )
    
    return demo

