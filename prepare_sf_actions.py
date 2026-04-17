"""Prepare per-entry action directories from a validation JSON file for
`generate_fast.py`.

For each entry in the JSON file we produce a directory named after the entry
``id`` under ``--work_dir`` containing:

    <id>/
      image.png            <- downloaded first_frame (from s3 or local path)
      poses.npy            <- (F+1, 4, 4) float32 camera-to-world trajectory in
                              OpenCV convention, synthesised from WASD/IJKL keys
                              using ``wan.utils.wasd_ijkl_to_c2ws.generate_and_save_trajectory``
      action.npy           <- (F, 4) int32 WASD per-frame keys (what cam-mode
                              fast inference loads as ``action.npy``)
      wasd_action.npy      <- same as action.npy (kept for act2cam compatibility)
      ijkl_action.npy      <- (F, 4) int32 IJKL per-frame keys
      intrinsics.npy       <- (F, 4) float32, constant [fx, fy, cx, cy]
      prompt.txt           <- prompt string
      meta.txt             <- "num_frames=<F>" (and other simple key=value fields)

The JSON schema (per entry) looks like::

    {
      "id": "the_dragon",
      "prompt": "...",
      "first_frame": "s3://bucket/key.png" | "/local/path.png",
      "actions": {
        "Movement": ["W", "WA", "-", ...],  # strings of W/A/S/D chars
        "Camera":   ["-", "\u2190", "\u2191\u2190", ...]  # arrow chars
      },
      "generation": {"height": 480, "width": 832, "num_frames": 81, ...}
    }

The ``Movement`` and ``Camera`` lists are two *independent* schedules that are
expanded to cover ``num_frames`` and then combined per-frame. When a schedule
is shorter than ``num_frames``, it is *tiled* (repeated end-to-end) by
default — e.g. ``["W"] * 7`` with ``num_frames=961`` becomes
``["W"] * 7 * 138`` truncated to 961. Pass ``--schedule_mode stretch`` to
fall back to the old "hold each token for ``num_frames/len`` frames" behavior.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

# allow "python prepare_sf_actions.py" to import from the local package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wan.utils.wasd_ijkl_to_c2ws import generate_and_save_trajectory  # noqa: E402


# --- action parsing -------------------------------------------------------

_WASD_INDEX = {"w": 0, "a": 1, "s": 2, "d": 3}
_IJKL_INDEX = {"i": 0, "j": 1, "k": 2, "l": 3}

# Camera arrows -> IJKL look-around keys.
#   ← look left  -> j (yaw left)
#   → look right -> l (yaw right)
#   ↑ look up    -> i (pitch up)
#   ↓ look down  -> k (pitch down)
_ARROW_TO_IJKL = {
    "\u2190": "j",   # ←
    "\u2192": "l",   # →
    "\u2191": "i",   # ↑
    "\u2193": "k",   # ↓
}


def _movement_token_to_wasd(token: str):
    """'WA' -> {'w', 'a'}; '-' or '' -> set()."""
    token = (token or "").strip()
    if not token or token == "-":
        return set()
    keys = set()
    for c in token.lower():
        if c in _WASD_INDEX:
            keys.add(c)
        # silently ignore unknown chars so the user can pass commas/spaces
    return keys


def _camera_token_to_ijkl(token: str):
    """'↑←' -> {'i', 'j'}; '-' or '' -> set()."""
    token = (token or "").strip()
    if not token or token == "-":
        return set()
    keys = set()
    for ch in token:
        if ch in _ARROW_TO_IJKL:
            keys.add(_ARROW_TO_IJKL[ch])
    return keys


def _expand_schedule_to_frames(schedule, num_frames, mode="repeat"):
    """Expand an arbitrary-length schedule to ``num_frames`` per-frame slots.

    Two modes are supported:

    * ``repeat`` (default): when the schedule is shorter than ``num_frames``
      the list is tiled end-to-end and then truncated, so e.g.
      ``["W"] * 7`` with ``num_frames=961`` becomes ``["W"] * 7 * 138`` and
      is sliced to length 961. Each original token consumes one frame. When
      the schedule is already at least ``num_frames`` long, the first
      ``num_frames`` entries are taken as-is (no stretching, no downsampling).
    * ``stretch`` (legacy): frame ``f`` takes
      ``schedule[floor(f * L / num_frames)]`` so each token is held for
      ``~num_frames / L`` frames regardless of the relative lengths.

    ``L == 0`` is always treated as "empty action on every frame".
    """
    L = len(schedule)
    if L == 0:
        return [set() for _ in range(num_frames)]

    if mode == "stretch":
        return [schedule[min(L - 1, int(f * L / num_frames))]
                for f in range(num_frames)]

    # "repeat" mode (default): tile then truncate.
    if L >= num_frames:
        return list(schedule[:num_frames])
    repeats = (num_frames + L - 1) // L  # ceil(num_frames / L)
    tiled = list(schedule) * repeats
    return tiled[:num_frames]


def build_wasd_ijkl(actions, num_frames, schedule_mode="repeat"):
    """Return (wasd_int32, ijkl_int32, frame_keys) given the JSON ``actions``."""
    mov_tokens = [_movement_token_to_wasd(t) for t in actions.get("Movement", [])]
    cam_tokens = [_camera_token_to_ijkl(t) for t in actions.get("Camera", [])]

    mov_per_frame = _expand_schedule_to_frames(
        mov_tokens, num_frames, mode=schedule_mode)
    cam_per_frame = _expand_schedule_to_frames(
        cam_tokens, num_frames, mode=schedule_mode)

    wasd = np.zeros((num_frames, 4), dtype=np.int32)
    ijkl = np.zeros((num_frames, 4), dtype=np.int32)
    frame_keys = []
    for f in range(num_frames):
        keys_f = []
        for c in mov_per_frame[f]:
            wasd[f, _WASD_INDEX[c]] = 1
            keys_f.append(c)
        for c in cam_per_frame[f]:
            ijkl[f, _IJKL_INDEX[c]] = 1
            keys_f.append(c)
        frame_keys.append(keys_f)
    return wasd, ijkl, frame_keys


# --- first-frame image handling ------------------------------------------

def fetch_first_frame(src: str, dst: str):
    """Copy/download ``src`` to ``dst``. Supports s3:// and local paths."""
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.startswith("s3://"):
        subprocess.check_call(["aws", "s3", "cp", "--quiet", src, dst])
    else:
        shutil.copyfile(src, dst)


# --- main ----------------------------------------------------------------

# Default intrinsics for 480 x 832 (same values as examples/03, examples/04).
#   [fx, fy, cx, cy]
_DEFAULT_INTRINSICS_480x832 = np.array(
    [415.5298, 415.6922, 415.77786, 239.77779], dtype=np.float32
)


# Long-video default: 961 frames ≈ 1 minute @ 16 FPS (README § "Tips").
DEFAULT_NUM_FRAMES = 961


def pad_frame_num_to_4n_plus_1(n: int) -> int:
    rem = (n - 1) % 4
    return n if rem == 0 else n + (4 - rem)


def prepare_entry(
    entry,
    work_dir,
    default_height=480,
    default_width=832,
    num_frames_override=None,
    default_num_frames=DEFAULT_NUM_FRAMES,
    schedule_mode="repeat",
):
    eid = entry["id"]
    out_dir = os.path.join(work_dir, eid)
    os.makedirs(out_dir, exist_ok=True)

    gen = entry.get("generation", {})
    # Priority: CLI override > per-entry JSON value > DEFAULT_NUM_FRAMES.
    # Action schedules ("Movement"/"Camera") are stretched to num_frames by
    # _expand_schedule_to_frames, so they are always "long enough" — each
    # schedule token just holds for ~num_frames / len(schedule) frames.
    if num_frames_override is not None:
        num_frames = int(num_frames_override)
    else:
        num_frames = int(gen.get("num_frames", default_num_frames))
    num_frames = pad_frame_num_to_4n_plus_1(num_frames)

    # --- image ---
    img_src = entry["first_frame"]
    ext = os.path.splitext(img_src)[1].lower() or ".png"
    img_dst = os.path.join(out_dir, f"image{ext}")
    fetch_first_frame(img_src, img_dst)

    # --- actions ---
    actions = entry.get("actions", {}) or {}
    wasd, ijkl, frame_keys = build_wasd_ijkl(
        actions, num_frames, schedule_mode=schedule_mode)

    # --- camera trajectory from keys ---
    # generate_and_save_trajectory returns (F+1, 4, 4) starting with identity.
    c2ws = np.array(generate_and_save_trajectory(frame_keys), dtype=np.float32)

    # --- intrinsics ---
    height = int(gen.get("height", default_height))
    width = int(gen.get("width", default_width))
    if (height, width) == (480, 832):
        K = _DEFAULT_INTRINSICS_480x832
    else:
        # scale cx, cy to target resolution (keep fx, fy proportional)
        sx = width / 832.0
        sy = height / 480.0
        K = np.array(
            [
                _DEFAULT_INTRINSICS_480x832[0] * sx,
                _DEFAULT_INTRINSICS_480x832[1] * sy,
                width / 2.0,
                height / 2.0,
            ],
            dtype=np.float32,
        )
    intrinsics = np.tile(K[None, :], (num_frames, 1)).astype(np.float32)

    # --- save ---
    np.save(os.path.join(out_dir, "poses.npy"), c2ws)
    np.save(os.path.join(out_dir, "action.npy"), wasd)          # wasd, cam-mode loader
    np.save(os.path.join(out_dir, "wasd_action.npy"), wasd)      # for act2cam compatibility
    np.save(os.path.join(out_dir, "ijkl_action.npy"), ijkl)
    np.save(os.path.join(out_dir, "intrinsics.npy"), intrinsics)

    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(entry.get("prompt", "").strip())

    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"id={eid}\n")
        f.write(f"num_frames={num_frames}\n")
        f.write(f"height={height}\n")
        f.write(f"width={width}\n")
        f.write(f"image={os.path.basename(img_dst)}\n")
        f.write(f"num_inference_steps={int(gen.get('num_inference_steps', 25))}\n")
        f.write(f"seed={int(gen.get('seed', 42))}\n")

    return {
        "id": eid,
        "dir": out_dir,
        "image": img_dst,
        "prompt": entry.get("prompt", "").strip(),
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="Path to validation JSON file.")
    parser.add_argument(
        "--work_dir",
        required=True,
        help="Directory to write per-entry action subdirectories into.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to write a TSV manifest (id\\tdir\\timage\\tnum_frames).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help=(
            "Override the per-entry num_frames for every entry (padded to "
            "4n+1). When unset, falls back to each entry's "
            "generation.num_frames, or DEFAULT_NUM_FRAMES "
            f"({DEFAULT_NUM_FRAMES}) if that field is missing."
        ),
    )
    parser.add_argument(
        "--default_num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=(
            "Fallback num_frames used only when an entry has no "
            "generation.num_frames field (default: "
            f"{DEFAULT_NUM_FRAMES}, i.e. ~1 min @ 16 FPS)."
        ),
    )
    parser.add_argument(
        "--schedule_mode",
        choices=["repeat", "stretch"],
        default="repeat",
        help=(
            "How to expand short action schedules to num_frames slots. "
            "'repeat' (default) tiles the schedule end-to-end (e.g. "
            "[W]*7 with num_frames=961 -> [W]*7 * 138 truncated to 961). "
            "'stretch' (legacy) holds each token for ~num_frames/len(schedule) "
            "frames, which can lead to very long holds when num_frames >> len."
        ),
    )
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    os.makedirs(args.work_dir, exist_ok=True)
    prepared = []
    for entry in data:
        info = prepare_entry(
            entry,
            args.work_dir,
            num_frames_override=args.num_frames,
            default_num_frames=args.default_num_frames,
            schedule_mode=args.schedule_mode,
        )
        prepared.append(info)
        print(
            f"[ok] {info['id']:45s} frames={info['num_frames']:4d} "
            f"dir={info['dir']}",
            flush=True,
        )

    manifest_path = args.manifest or os.path.join(args.work_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("id\tdir\timage\tnum_frames\theight\twidth\n")
        for info in prepared:
            f.write(
                "{id}\t{dir}\t{image}\t{num_frames}\t{height}\t{width}\n".format(**info)
            )
    print(f"[done] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
