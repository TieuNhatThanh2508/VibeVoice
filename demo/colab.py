# %%writefile /content/VibeVoice/demo/colab.py
# Original Code: https://github.com/microsoft/VibeVoice/blob/main/demo/gradio_demo.py
"""
VibeVoice Gradio Demo - Refactored with modular structure
"""

import torch
import gradio as gr
import click
from transformers import set_seed

# Import modules
from demo.colab_modules import (
    config,
    ModelManager,
    VoiceManager,
    AudioProcessor,
    PodcastGenerator,
    create_demo_interface,
    create_prompt_builder_ui
)
from demo.colab_utils import download_model


def main(model_path=None, inference_steps=None, debug=False, share=False):
    """
    Main function to launch the demo
    
    Args:
        model_path: HuggingFace model repo ID or path
        inference_steps: Number of inference steps
        debug: Enable debug mode
        share: Enable sharing
    """
    # Use config defaults if not provided
    if model_path is None:
        model_path = config.model.default_model_path
    
    if inference_steps is None:
        inference_steps = config.model.default_inference_steps
    
    # Download model if needed
    model_folder = download_model(
        model_path,
        download_folder=config.file.download_folder,
        redownload=config.file.redownload
    )
    
    # Determine device
    device = config.model.default_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("⚠️ CUDA not available, using CPU")
    
    # Set seed for reproducibility
    set_seed(42)
    
    print("🎙️ Initializing VibeVoice ...")
    
    # Initialize managers
    model_manager = ModelManager(
        model_path=model_folder,
        device=device,
        inference_steps=inference_steps
    )
    model_manager.load_model()
    
    voice_manager = VoiceManager()
    audio_processor = AudioProcessor()
    
    # Create generator
    generator = PodcastGenerator(
        model_manager=model_manager,
        voice_manager=voice_manager,
        audio_processor=audio_processor
    )
    
    # Create UI interfaces
    demo1 = create_demo_interface(generator, voice_manager, audio_processor)
    demo2 = create_prompt_builder_ui()
    
    # Create tabbed interface
    demo = gr.TabbedInterface(
        [demo1, demo2],
        [config.ui.tab1_title, config.ui.tab2_title],
        title="",
        theme=getattr(gr.themes, config.ui.theme)(),
        css=config.ui.custom_css
    )
    
    print("🚀 Launching Gradio Demo...")
    demo.queue().launch(debug=debug, share=share)


@click.command()
@click.option(
    "--model_path",
    default=None,
    help="Hugging Face Model Repo ID. Defaults to config."
)
@click.option(
    "--inference_steps",
    default=None,
    show_default=True,
    type=int,
    help="Number of inference steps for generation. Defaults to config."
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode."
)
@click.option(
    "--share",
    is_flag=True,
    default=False,
    help="Enable sharing of the interface."
)
def cli_main(model_path, inference_steps, debug, share):
    """CLI entry point"""
    main(
        model_path=model_path,
        inference_steps=inference_steps,
        debug=debug,
        share=share
    )


if __name__ == "__main__":
    cli_main()
