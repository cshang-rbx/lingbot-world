"""
LingBot-World Gradio Demo (Multi-GPU)
A web interface for generating videos from images using camera control.

Launch with: torchrun --nproc_per_node=8 app.py
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

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video
from wan.distributed.util import init_distributed_group

# Configure logging (only rank 0 logs info)
RANK = int(os.getenv("RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))

if RANK == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )
else:
    logging.basicConfig(level=logging.ERROR)

# Default checkpoint directory
CKPT_DIR = "lingbot-world-base-cam"

# Global model instance
model = None


def setup_distributed():
    """Initialize distributed training environment."""
    global RANK, WORLD_SIZE, LOCAL_RANK
    
    if WORLD_SIZE > 1:
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=RANK,
            world_size=WORLD_SIZE
        )
        # Initialize sequence parallel groups
        init_distributed_group()
        logging.info(f"Distributed initialized: rank={RANK}, world_size={WORLD_SIZE}, local_rank={LOCAL_RANK}")
    else:
        logging.info("Running in single-GPU mode")


def load_model():
    """Load the WanI2V model with multi-GPU support."""
    global model
    
    if model is not None:
        return model
    
    logging.info(f"[Rank {RANK}] Loading LingBot-World model...")
    cfg = WAN_CONFIGS["i2v-A14B"]
    
    # Enable FSDP and sequence parallelism for multi-GPU
    use_fsdp = WORLD_SIZE > 1
    use_sp = WORLD_SIZE > 1
    
    model = wan.WanI2V(
        config=cfg,
        checkpoint_dir=CKPT_DIR,
        device_id=LOCAL_RANK,
        rank=RANK,
        t5_fsdp=use_fsdp,
        dit_fsdp=use_fsdp,
        use_sp=use_sp,
        t5_cpu=False,
        init_on_cpu=False,  # With FSDP, we don't init on CPU
    )
    
    logging.info(f"[Rank {RANK}] Model loaded successfully!")
    
    if dist.is_initialized():
        dist.barrier()
    
    return model


def sync_generation_params(image_bytes, prompt, action_path, size, frame_num, seed, guide_scale, sample_steps, shift):
    """Synchronize generation parameters across all ranks."""
    if not dist.is_initialized():
        return image_bytes, prompt, action_path, size, frame_num, seed, guide_scale, sample_steps, shift
    
    # Pack parameters into a list for broadcasting
    if RANK == 0:
        params = [image_bytes, prompt, action_path, size, frame_num, seed, guide_scale, sample_steps, shift]
    else:
        params = [None] * 9
    
    dist.broadcast_object_list(params, src=0)
    
    return tuple(params)


def generate_video_distributed(
    image: Image.Image,
    prompt: str,
    intrinsics_file,
    poses_file,
    resolution: str,
    frame_num: int,
    seed: int,
    guide_scale: float,
    sample_steps: int,
):
    """
    Generate video from input image and prompt (multi-GPU).
    Only rank 0 handles inputs/outputs, all ranks participate in generation.
    """
    import io
    
    # Prepare parameters on rank 0
    image_bytes = None
    action_path = None
    temp_action_dir = None
    
    if RANK == 0:
        if image is None:
            raise ValueError("Please upload an input image!")
        
        if not prompt or prompt.strip() == "":
            raise ValueError("Please enter a text prompt!")
        
        # Validate frame_num (should be 4n+1)
        if (frame_num - 1) % 4 != 0:
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            logging.warning(f"Frame number adjusted to {frame_num} (must be 4n+1)")
        
        # Serialize image to bytes for broadcasting
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        image_bytes = buf.getvalue()
        
        # Setup action path if camera files are provided
        if intrinsics_file is not None and poses_file is not None:
            temp_action_dir = tempfile.mkdtemp()
            intrinsics_path = os.path.join(temp_action_dir, "intrinsics.npy")
            poses_path = os.path.join(temp_action_dir, "poses.npy")
            
            intrinsics = np.load(intrinsics_file)
            poses = np.load(poses_file)
            np.save(intrinsics_path, intrinsics)
            np.save(poses_path, poses)
            
            action_path = temp_action_dir
            logging.info(f"Using camera controls: intrinsics={intrinsics.shape}, poses={poses.shape}")
    
    # Get size configuration
    if resolution == "480P":
        size = "480*832"
        shift = 3.0
    else:  # 720P
        size = "720*1280"
        shift = 5.0
    
    # Synchronize parameters across all ranks
    image_bytes, prompt, action_path, size, frame_num, seed, guide_scale, sample_steps, shift = \
        sync_generation_params(image_bytes, prompt, action_path, size, frame_num, seed, guide_scale, sample_steps, shift)
    
    # Reconstruct image from bytes on all ranks
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Sync seed across ranks
    if dist.is_initialized():
        seed_tensor = torch.tensor([seed if seed >= 0 else 42], device=f"cuda:{LOCAL_RANK}")
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    
    logging.info(f"[Rank {RANK}] Generating video: {resolution}, {frame_num} frames, seed={seed}")
    
    # Load model (already loaded, just get reference)
    wan_model = load_model()
    
    try:
        # Generate video (all ranks participate)
        video = wan_model.generate(
            input_prompt=prompt,
            img=img,
            action_path=action_path,
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,
            shift=shift,
            sample_solver='unipc',
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=False,  # Keep in memory for multi-GPU
        )
        
        output_path = None
        
        # Only rank 0 saves the video
        if RANK == 0 and video is not None:
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
            output_path = str(output_path)
        
        # Cleanup
        if video is not None:
            del video
        gc.collect()
        torch.cuda.empty_cache()
        
        if dist.is_initialized():
            dist.barrier()
        
        return output_path
        
    finally:
        # Cleanup temp directory on rank 0
        if RANK == 0 and temp_action_dir is not None:
            import shutil
            shutil.rmtree(temp_action_dir, ignore_errors=True)


def worker_loop():
    """
    Worker loop for non-rank-0 processes.
    Waits for generation signals and participates in distributed inference.
    """
    import io
    
    logging.info(f"[Rank {RANK}] Worker started, waiting for generation tasks...")
    
    wan_model = load_model()
    
    while True:
        # Wait for signal from rank 0
        signal = [None]
        dist.broadcast_object_list(signal, src=0)
        
        if signal[0] == "EXIT":
            logging.info(f"[Rank {RANK}] Received exit signal, shutting down...")
            break
        elif signal[0] == "GENERATE":
            # Receive generation parameters
            params = [None] * 9
            dist.broadcast_object_list(params, src=0)
            image_bytes, prompt, action_path, size, frame_num, seed, guide_scale, sample_steps, shift = params
            
            # Reconstruct image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Sync seed
            seed_tensor = torch.tensor([seed], device=f"cuda:{LOCAL_RANK}")
            dist.broadcast(seed_tensor, src=0)
            seed = int(seed_tensor.item())
            
            logging.info(f"[Rank {RANK}] Generating video...")
            
            # Participate in generation
            video = wan_model.generate(
                input_prompt=prompt,
                img=img,
                action_path=action_path,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=frame_num,
                shift=shift,
                sample_solver='unipc',
                sampling_steps=sample_steps,
                guide_scale=guide_scale,
                seed=seed,
                offload_model=False,
            )
            
            # Cleanup
            if video is not None:
                del video
            gc.collect()
            torch.cuda.empty_cache()
            
            dist.barrier()
            logging.info(f"[Rank {RANK}] Generation complete")


def generate_video_wrapper(image, prompt, intrinsics_file, poses_file, resolution, frame_num, seed, guide_scale, sample_steps):
    """Wrapper for Gradio that signals workers and coordinates generation."""
    import io
    
    try:
        if image is None:
            raise ValueError("Please upload an input image!")
        if not prompt or prompt.strip() == "":
            raise ValueError("Please enter a text prompt!")
        
        # Validate frame_num
        if (frame_num - 1) % 4 != 0:
            frame_num = ((frame_num - 1) // 4) * 4 + 1
        
        # Prepare image bytes
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        image_bytes = buf.getvalue()
        
        # Setup action path
        action_path = None
        temp_action_dir = None
        
        if intrinsics_file is not None and poses_file is not None:
            temp_action_dir = tempfile.mkdtemp()
            intrinsics_path = os.path.join(temp_action_dir, "intrinsics.npy")
            poses_path = os.path.join(temp_action_dir, "poses.npy")
            
            intrinsics = np.load(intrinsics_file)
            poses = np.load(poses_file)
            np.save(intrinsics_path, intrinsics)
            np.save(poses_path, poses)
            action_path = temp_action_dir
        
        # Size config
        if resolution == "480P":
            size = "480*832"
            shift = 3.0
        else:
            size = "720*1280"
            shift = 5.0
        
        # Signal workers if distributed
        if dist.is_initialized() and WORLD_SIZE > 1:
            signal = ["GENERATE"]
            dist.broadcast_object_list(signal, src=0)
            
            params = [image_bytes, prompt, action_path, size, frame_num, seed, guide_scale, sample_steps, shift]
            dist.broadcast_object_list(params, src=0)
        
        # Reconstruct image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Sync seed
        if dist.is_initialized():
            seed_tensor = torch.tensor([seed if seed >= 0 else 42], device=f"cuda:{LOCAL_RANK}")
            dist.broadcast(seed_tensor, src=0)
            seed = int(seed_tensor.item())
        
        logging.info(f"Generating: {resolution}, {frame_num} frames, seed={seed}")
        
        wan_model = load_model()
        
        video = wan_model.generate(
            input_prompt=prompt,
            img=img,
            action_path=action_path,
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,
            shift=shift,
            sample_solver='unipc',
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=(WORLD_SIZE == 1),  # Only offload in single-GPU mode
        )
        
        output_path = None
        
        if video is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"lingbot_world_{timestamp}.mp4")
            
            cfg = WAN_CONFIGS["i2v-A14B"]
            save_video(
                tensor=video[None],
                save_file=output_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            logging.info(f"Video saved to {output_path}")
        
        # Cleanup
        if video is not None:
            del video
        gc.collect()
        torch.cuda.empty_cache()
        
        if dist.is_initialized():
            dist.barrier()
        
        if temp_action_dir:
            import shutil
            shutil.rmtree(temp_action_dir, ignore_errors=True)
        
        return output_path
        
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise


def load_example(example_name: str):
    """Load an example from the examples directory."""
    example_path = Path("examples") / example_name
    
    if not example_path.exists():
        return None, "", None, None
    
    image_path = example_path / "image.jpg"
    image = Image.open(image_path) if image_path.exists() else None
    
    prompt_path = example_path / "prompt.txt"
    prompt = prompt_path.read_text().strip() if prompt_path.exists() else ""
    
    intrinsics_path = example_path / "intrinsics.npy"
    poses_path = example_path / "poses.npy"
    
    intrinsics = str(intrinsics_path) if intrinsics_path.exists() else None
    poses = str(poses_path) if poses_path.exists() else None
    
    return image, prompt, intrinsics, poses


# Custom CSS
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

.gpu-badge {
    display: inline-block;
    background: rgba(0, 255, 200, 0.15);
    border: 1px solid var(--primary-color);
    color: var(--primary-color);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-left: 0.5rem;
}

footer {
    display: none !important;
}
"""


def create_gradio_interface():
    """Create and return the Gradio interface."""
    import gradio as gr
    
    gpu_info = f"{WORLD_SIZE} GPU{'s' if WORLD_SIZE > 1 else ''}"
    
    with gr.Blocks(css=custom_css, title="LingBot-World", theme=gr.themes.Base()) as demo:
        
        gr.HTML(f"""
            <div class="main-header">
                <h1>🌍 LingBot-World</h1>
                <p>Open-source World Simulator • Image to Video Generation 
                <span class="gpu-badge">⚡ {gpu_info}</span></p>
            </div>
        """)
        
        with gr.Row():
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
        
        # Example loaders
        example_00_btn.click(fn=lambda: load_example("00"), outputs=[input_image, prompt, intrinsics_file, poses_file])
        example_01_btn.click(fn=lambda: load_example("01"), outputs=[input_image, prompt, intrinsics_file, poses_file])
        example_02_btn.click(fn=lambda: load_example("02"), outputs=[input_image, prompt, intrinsics_file, poses_file])
        
        # Generate button
        generate_btn.click(
            fn=generate_video_wrapper,
            inputs=[input_image, prompt, intrinsics_file, poses_file, resolution, frame_num, seed, guide_scale, sample_steps],
            outputs=output_video,
        )
        
        gr.HTML(f"""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: var(--text-secondary); font-family: 'Space Mono', monospace; font-size: 0.8rem;">
                <p>LingBot-World by Robbyant Team • Running on {gpu_info}</p>
                <p>Launch: <code style="color: var(--primary-color);">torchrun --nproc_per_node={WORLD_SIZE} app.py</code></p>
            </div>
        """)
    
    return demo


def main():
    """Main entry point."""
    print("=" * 60)
    print("🌍 LingBot-World Gradio Demo (Multi-GPU)")
    print(f"   Rank: {RANK} / {WORLD_SIZE}, Local GPU: {LOCAL_RANK}")
    print("=" * 60)
    
    # Initialize distributed
    setup_distributed()
    
    # Check checkpoint directory
    if RANK == 0 and not os.path.exists(CKPT_DIR):
        print(f"\n⚠️  Warning: Checkpoint directory '{CKPT_DIR}' not found!")
        print("   Please download the model first:")
        print("   huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam")
    
    # Pre-load model on all ranks
    load_model()
    
    if RANK == 0:
        # Rank 0 runs the Gradio server
        import gradio as gr
        demo = create_gradio_interface()
        demo.queue(max_size=1)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True,
            prevent_thread_lock=False,
        )
    else:
        # Other ranks wait in worker loop
        worker_loop()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
torchrun --nproc_per_node=8 --master_port=29501 app.py
"""