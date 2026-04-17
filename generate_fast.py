import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
            "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
        "image":
            "examples/04/image.jpg",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # Batch-manifest mode skips per-input validation; per-row prompt/image/
    # action_path come from the manifest instead.
    if not args.manifest:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
            args.image = EXAMPLE_PROMPT[args.task]["image"]

        if args.task == "i2v-A14B":
            assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--max_attention_size",
        type=int,
        default=None,
        help="The size of kv cache during inference.")
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help=(
            "Number of leading latent frames to keep pinned as an attention "
            "sink when the self-attn KV cache rolls. Only has a visible "
            "effect when --max_attention_size is also set (which activates "
            "the rolling gate by mirroring its value into local_attn_size) "
            "AND the allocated cache is smaller than the full sequence. "
            "1 latent frame == 4 pixel frames == 0.25 s at 16 FPS."
        ))
    parser.add_argument(
        "--save_dir",
        type=str,
        default='output',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Optional TSV manifest (as produced by prepare_sf_actions.py). "
            "When set, the pipeline is loaded once and each row is generated "
            "in sequence. Expected columns: id, dir, image, num_frames, "
            "height, width. Per-row prompt is read from <dir>/prompt.txt."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        type=str2bool,
        default=True,
        help="Skip rows whose output mp4 already exists (batch mode only).",
    )

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def _load_manifest(manifest_path):
    """Load a TSV manifest produced by prepare_sf_actions.py.

    Returns a list of dicts keyed by the header columns. A header row is
    required.
    """
    rows = []
    with open(manifest_path) as f:
        header_line = f.readline()
        if not header_line:
            return rows
        header = [h.strip() for h in header_line.rstrip("\n").split("\t")]
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            # Pad short rows with empty strings so zip() doesn't silently drop
            # columns when a field is missing.
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            row = dict(zip(header, parts))
            if not row.get("id"):
                continue
            rows.append(row)
    return rows


def _read_prompt(row):
    """Return the prompt for a manifest row, preferring <dir>/prompt.txt."""
    prompt_path = os.path.join(row["dir"], "prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path) as f:
            return f.read().strip()
    return row.get("prompt", "").strip()


def _save_video_rank0(video, save_file, cfg, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving generated video to {save_file}")
    save_video(
        tensor=video[None],
        save_file=save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))


def _broadcast_obj(obj, src=0):
    """Broadcast a picklable object from ``src`` to all ranks."""
    if not dist.is_initialized():
        return obj
    holder = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def _run_one(wan_i2v, cfg, args, *, prompt, image_path, action_path,
             frame_num, save_file, rank):
    """Generate a single video and (on rank 0) save it to ``save_file``."""
    logging.info(f"Input prompt: {prompt}")
    logging.info(f"Input image: {image_path}")
    logging.info(f"Action path: {action_path}")
    logging.info(f"Frames: {frame_num} -> {save_file}")

    img = Image.open(image_path).convert("RGB")
    video = wan_i2v.generate(
        prompt,
        img,
        action_path=action_path,
        chunk_size=3,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=frame_num,
        shift=args.sample_shift,
        seed=args.base_seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size,
        sink_size=args.sink_size)

    if rank == 0:
        _save_video_rank0(video, save_file, cfg, args.save_dir)
    del video
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ---- Load pipeline once ---------------------------------------------
    logging.info("Creating WanI2VFast pipeline.")
    wan_i2v = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    # ---- Batch (manifest) mode ------------------------------------------
    if args.manifest:
        # Rank 0 reads the manifest and broadcasts it so every rank iterates
        # over the exact same row order.
        if rank == 0:
            rows = _load_manifest(args.manifest)
            logging.info(
                f"Loaded manifest {args.manifest} with {len(rows)} entries.")
        else:
            rows = None
        rows = _broadcast_obj(rows, src=0)

        os.makedirs(args.save_dir, exist_ok=True)
        total = len(rows)
        for idx, row in enumerate(rows):
            eid = row["id"]
            save_file = args.save_file or os.path.join(args.save_dir, f"{eid}.mp4")
            # For batch mode we always name by id regardless of --save_file
            # (which only applies to single-item mode).
            save_file = os.path.join(args.save_dir, f"{eid}.mp4")

            if args.skip_existing and os.path.exists(save_file):
                if rank == 0:
                    logging.info(
                        f"[skip {idx+1}/{total}] {save_file} already exists")
                continue

            action_path = row["dir"]
            image_path = row["image"]
            try:
                frame_num = int(row.get("num_frames") or args.frame_num)
            except (TypeError, ValueError):
                frame_num = args.frame_num

            prompt = _read_prompt(row)
            if rank == 0:
                logging.info(
                    f"[run {idx+1}/{total}] id={eid} frames={frame_num}")

            _run_one(
                wan_i2v, cfg, args,
                prompt=prompt,
                image_path=image_path,
                action_path=action_path,
                frame_num=frame_num,
                save_file=save_file,
                rank=rank,
            )

        if rank == 0:
            logging.info(f"[done] processed {total} entries -> {args.save_dir}")
    else:
        # ---- Single-input mode (backward compatible) --------------------
        logging.info(f"Input prompt: {args.prompt}")
        img = None
        if args.image is not None:
            img = Image.open(args.image).convert("RGB")
            logging.info(f"Input image: {args.image}")

        # prompt extend (rank-0 broadcast kept for parity with upstream)
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                input_prompt = [args.prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            action_path=args.action_path,
            chunk_size=3,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            seed=args.base_seed,
            offload_model=args.offload_model,
            max_attention_size=args.max_attention_size,
            sink_size=args.sink_size)

        if rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            if args.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                         "_")[:50]
                suffix = '.mp4'
                args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
                args.save_file = f'{args.save_dir}/{args.save_file}'

            logging.info(f"Saving generated video to {args.save_file}")
            save_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            if "s2v" in args.task:
                if args.enable_tts is False:
                    merge_video_audio(video_path=args.save_file, audio_path=args.audio)
                else:
                    merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
        del video

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
