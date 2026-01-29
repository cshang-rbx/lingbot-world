"""
LingBot-World Gradio Demo
A web interface for generating videos from images using camera control.
"""

import gc
import logging
import os
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import gradio as gr
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# Default checkpoint directory
CKPT_DIR = "lingbot-world-base-cam"

# Global model instance (loaded once)
model = None


def load_model():
    """Load the WanI2V model (singleton pattern)."""
    global model
    if model is None:
        logging.info("Loading LingBot-World model...")
        cfg = WAN_CONFIGS["i2v-A14B"]
        model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=CKPT_DIR,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=True,
        )
        logging.info("Model loaded successfully!")
    return model


def generate_video(
    image: Image.Image,
    prompt: str,
    intrinsics_file,
    poses_file,
    resolution: str,
    frame_num: int,
    seed: int,
    guide_scale: float,
    sample_steps: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate video from input image and prompt.
    
    Args:
        image: Input PIL Image
        prompt: Text prompt describing the video
        intrinsics_file: Optional camera intrinsics .npy file
        poses_file: Optional camera poses .npy file
        resolution: Video resolution (480P or 720P)
        frame_num: Number of frames to generate (must be 4n+1)
        seed: Random seed (-1 for random)
        guide_scale: Classifier-free guidance scale
        sample_steps: Number of sampling steps
    
    Returns:
        Path to generated video file
    """
    if image is None:
        raise gr.Error("Please upload an input image!")
    
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter a text prompt!")
    
    # Validate frame_num (should be 4n+1)
    if (frame_num - 1) % 4 != 0:
        frame_num = ((frame_num - 1) // 4) * 4 + 1
        logging.warning(f"Frame number adjusted to {frame_num} (must be 4n+1)")
    
    # Get size configuration
    if resolution == "480P":
        size = "480*832"
        shift = 3.0
    else:  # 720P
        size = "720*1280"
        shift = 5.0
    
    # Setup action path if camera files are provided
    action_path = None
    temp_action_dir = None
    
    if intrinsics_file is not None and poses_file is not None:
        # Create temporary directory for camera files
        temp_action_dir = tempfile.mkdtemp()
        
        # Copy uploaded files to temp directory
        intrinsics_path = os.path.join(temp_action_dir, "intrinsics.npy")
        poses_path = os.path.join(temp_action_dir, "poses.npy")
        
        # Load and save the numpy files
        intrinsics = np.load(intrinsics_file)
        poses = np.load(poses_file)
        np.save(intrinsics_path, intrinsics)
        np.save(poses_path, poses)
        
        action_path = temp_action_dir
        logging.info(f"Using camera controls from uploaded files")
        logging.info(f"  Intrinsics shape: {intrinsics.shape}")
        logging.info(f"  Poses shape: {poses.shape}")
    
    # Load model
    wan_model = load_model()
    
    # Convert image to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    logging.info(f"Generating video...")
    logging.info(f"  Resolution: {resolution} ({size})")
    logging.info(f"  Frame count: {frame_num}")
    logging.info(f"  Seed: {seed}")
    logging.info(f"  Guide scale: {guide_scale}")
    logging.info(f"  Sample steps: {sample_steps}")
    logging.info(f"  Prompt: {prompt[:100]}...")
    
    try:
        # Generate video
        video = wan_model.generate(
            input_prompt=prompt,
            img=image,
            action_path=action_path,
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,
            shift=shift,
            sample_solver='unipc',
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            seed=seed if seed >= 0 else -1,
            offload_model=True,
        )
        
        # Save video to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"lingbot_world_{timestamp}.mp4"
        
        cfg = WAN_CONFIGS["i2v-A14B"]
        save_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        logging.info(f"Video saved to {output_path}")
        
        # Cleanup
        del video
        gc.collect()
        torch.cuda.empty_cache()
        
        return str(output_path)
        
    finally:
        # Cleanup temp directory
        if temp_action_dir is not None:
            import shutil
            shutil.rmtree(temp_action_dir, ignore_errors=True)


def load_example(example_name: str):
    """Load an example from the examples directory."""
    example_path = Path("examples") / example_name
    
    if not example_path.exists():
        return None, "", None, None
    
    # Load image
    image_path = example_path / "image.jpg"
    image = Image.open(image_path) if image_path.exists() else None
    
    # Load prompt
    prompt_path = example_path / "prompt.txt"
    prompt = ""
    if prompt_path.exists():
        prompt = prompt_path.read_text().strip()
    
    # Return file paths for numpy files
    intrinsics_path = example_path / "intrinsics.npy"
    poses_path = example_path / "poses.npy"
    
    intrinsics = str(intrinsics_path) if intrinsics_path.exists() else None
    poses = str(poses_path) if poses_path.exists() else None
    
    return image, prompt, intrinsics, poses


# Custom CSS for a distinctive, modern aesthetic
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
    --primary-color: #00ffc8;
    --secondary-color: #ff6b35;
    --bg-dark: #0a0a0f;
    --bg-card: #12121a;
    --text-primary: #e8e8e8;
    --text-secondary: #888;
    --border-color: #2a2a35;
    --accent-gradient: linear-gradient(135deg, #00ffc8 0%, #00a8ff 50%, #ff6b35 100%);
}

.gradio-container {
    font-family: 'Syne', sans-serif !important;
    background: var(--bg-dark) !important;
    background-image: 
        radial-gradient(ellipse at 20% 20%, rgba(0, 255, 200, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(255, 107, 53, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0, 168, 255, 0.04) 0%, transparent 70%);
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 1rem;
}

.main-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    font-size: 3rem;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

.main-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-family: 'Space Mono', monospace;
}

.gr-panel, .gr-box, .gr-form {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}

.gr-input, .gr-textarea, .gr-dropdown {
    background: var(--bg-dark) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
}

.gr-input:focus, .gr-textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 200, 0.15) !important;
}

.gr-button-primary {
    background: var(--accent-gradient) !important;
    border: none !important;
    color: var(--bg-dark) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.8rem 2rem !important;
    border-radius: 8px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 255, 200, 0.25) !important;
}

.gr-button-secondary {
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.gr-button-secondary:hover {
    border-color: var(--primary-color) !important;
    color: var(--primary-color) !important;
}

label {
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.gr-slider input[type="range"] {
    accent-color: var(--primary-color) !important;
}

.section-title {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700;
    font-size: 1.2rem;
    color: var(--primary-color);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}

.gr-accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}

.example-btn {
    background: rgba(0, 255, 200, 0.1) !important;
    border: 1px solid rgba(0, 255, 200, 0.3) !important;
    color: var(--primary-color) !important;
}

.example-btn:hover {
    background: rgba(0, 255, 200, 0.2) !important;
}

.info-text {
    color: var(--text-secondary);
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

footer {
    display: none !important;
}
"""

# Build Gradio interface
with gr.Blocks(css=custom_css, title="LingBot-World", theme=gr.themes.Base()) as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🌍 LingBot-World</h1>
            <p>Open-source World Simulator • Image to Video Generation</p>
        </div>
    """)
    
    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=1):
            gr.HTML('<div class="section-title">📷 Input</div>')
            
            input_image = gr.Image(
                label="Input Image",
                type="pil",
                height=300,
            )
            
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the video you want to generate...",
                lines=4,
                max_lines=8,
            )
            
            with gr.Accordion("🎥 Camera Control (Optional)", open=False):
                gr.HTML('<p class="info-text">Upload camera intrinsics and poses for controlled camera movement.</p>')
                intrinsics_file = gr.File(
                    label="intrinsics.npy",
                    file_types=[".npy"],
                )
                poses_file = gr.File(
                    label="poses.npy", 
                    file_types=[".npy"],
                )
            
            with gr.Accordion("⚙️ Generation Settings", open=True):
                resolution = gr.Radio(
                    choices=["480P", "720P"],
                    value="480P",
                    label="Resolution",
                )
                
                frame_num = gr.Slider(
                    minimum=17,
                    maximum=257,
                    value=81,
                    step=4,
                    label="Frame Count (4n+1)",
                    info="Number of frames to generate",
                )
                
                seed = gr.Number(
                    value=42,
                    label="Seed",
                    info="Use -1 for random seed",
                    precision=0,
                )
                
            with gr.Accordion("🔧 Advanced Settings", open=False):
                guide_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.5,
                    label="Guidance Scale",
                    info="Higher values = stronger prompt adherence",
                )
                
                sample_steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=40,
                    step=5,
                    label="Sampling Steps",
                    info="More steps = better quality but slower",
                )
            
            generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
        
        # Right column - Output
        with gr.Column(scale=1):
            gr.HTML('<div class="section-title">🎬 Output</div>')
            
            output_video = gr.Video(
                label="Generated Video",
                height=400,
            )
            
            gr.HTML('<div class="section-title" style="margin-top: 1.5rem;">📁 Examples</div>')
            gr.HTML('<p class="info-text">Click an example to load it into the interface.</p>')
            
            with gr.Row():
                example_00_btn = gr.Button("Example 00", variant="secondary", elem_classes=["example-btn"])
                example_01_btn = gr.Button("Example 01", variant="secondary", elem_classes=["example-btn"])
                example_02_btn = gr.Button("Example 02", variant="secondary", elem_classes=["example-btn"])
    
    # Example loading functions
    def load_example_00():
        return load_example("00")
    
    def load_example_01():
        return load_example("01")
    
    def load_example_02():
        return load_example("02")
    
    # Connect example buttons
    example_00_btn.click(
        fn=load_example_00,
        outputs=[input_image, prompt, intrinsics_file, poses_file]
    )
    example_01_btn.click(
        fn=load_example_01,
        outputs=[input_image, prompt, intrinsics_file, poses_file]
    )
    example_02_btn.click(
        fn=load_example_02,
        outputs=[input_image, prompt, intrinsics_file, poses_file]
    )
    
    # Connect generate button
    generate_btn.click(
        fn=generate_video,
        inputs=[
            input_image,
            prompt,
            intrinsics_file,
            poses_file,
            resolution,
            frame_num,
            seed,
            guide_scale,
            sample_steps,
        ],
        outputs=output_video,
    )
    
    # Footer info
    gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: var(--text-secondary); font-family: 'Space Mono', monospace; font-size: 0.8rem;">
            <p>LingBot-World by Robbyant Team • Built on Wan2.2</p>
            <p>For multi-GPU inference, use the CLI: <code style="color: var(--primary-color);">torchrun --nproc_per_node=8 generate.py</code></p>
        </div>
    """)


if __name__ == "__main__":
    # Pre-load model
    print("=" * 60)
    print("🌍 LingBot-World Gradio Demo")
    print("=" * 60)
    
    # Check if checkpoint directory exists
    if not os.path.exists(CKPT_DIR):
        print(f"\n⚠️  Warning: Checkpoint directory '{CKPT_DIR}' not found!")
        print("   Please download the model first:")
        print("   huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam")
        print("\n   Starting demo anyway (will error on generation)...")
    
    # Launch Gradio app
    demo.queue(max_size=1)  # Queue with max 1 concurrent request
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
