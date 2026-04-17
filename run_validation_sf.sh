#!/usr/bin/env bash
#
# End-to-end driver that takes a validation JSON file (same schema as
# validation_sf.json) and produces one .mp4 per entry under ``$OUTPUT_DIR``.
#
# The model weights are loaded ONCE and every entry in the manifest is
# generated in the same process.
#
# Usage:
#   bash run_validation_sf.sh <weight_dir> <output_dir> [json_path]
#
# Example:
#   NUM_FRAMES=481 NPROC=4 bash run_validation_sf.sh lingbot-world-base-cam \
#        /home/builder/workspace/video_eval/manual/lingbot-world-fast/sf_validation_fast \
#        validation_sf.json

set -eo pipefail

WEIGHT_DIR=${1:?weight_dir required (e.g. lingbot-world-base-cam)}
OUTPUT_DIR=${2:?output_dir required}
JSON_PATH=${3:-"$(dirname "$0")/validation_sf.json"}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORK_DIR="${OUTPUT_DIR}/actions"
LOG_DIR="${OUTPUT_DIR}/logs"
MANIFEST="${WORK_DIR}/manifest.tsv"

mkdir -p "$OUTPUT_DIR" "$WORK_DIR" "$LOG_DIR"

# 1) Build per-entry action dirs (poses.npy / action.npy / intrinsics.npy /
#    image.png / prompt.txt / meta.txt) and the manifest.tsv file.
#
# NUM_FRAMES (optional) forces the same length for every entry, ignoring
# per-entry generation.num_frames. Defaults to prepare_sf_actions.py's
# DEFAULT_NUM_FRAMES (961 ≈ 1 minute @ 16 FPS) when the JSON entries do
# not carry the field. Set e.g. ``NUM_FRAMES=961`` to force all entries.
PREPARE_ARGS=(--json "$JSON_PATH" --work_dir "$WORK_DIR" --manifest "$MANIFEST")
if [ -n "${NUM_FRAMES:-}" ]; then
    PREPARE_ARGS+=(--num_frames "$NUM_FRAMES")
fi
# SCHEDULE_MODE: "repeat" (default) tiles short action schedules to cover
# num_frames; "stretch" holds each token for num_frames/len(schedule) frames.
if [ -n "${SCHEDULE_MODE:-}" ]; then
    PREPARE_ARGS+=(--schedule_mode "$SCHEDULE_MODE")
fi
echo "[run] preparing action dirs under: $WORK_DIR"
python "$SCRIPT_DIR/prepare_sf_actions.py" "${PREPARE_ARGS[@]}"

# 2) Run generate_fast.py ONCE over the whole manifest. Weights are loaded
#    a single time and every entry is produced in sequence inside the same
#    distributed process group.
NPROC=${NPROC:-8}
SIZE=${SIZE:-"480*832"}
ULYSSES_SIZE=${ULYSSES_SIZE:-$NPROC}
BATCH_LOG="${LOG_DIR}/batch.log"

echo "[run] generating ${OUTPUT_DIR} from manifest ${MANIFEST} (nproc=${NPROC})"
set -x
torchrun --nproc_per_node="$NPROC" "$SCRIPT_DIR/generate_fast.py" \
    --task i2v-A14B \
    --size "$SIZE" \
    --ckpt_dir "$WEIGHT_DIR" \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size "$ULYSSES_SIZE" \
    --save_dir "$OUTPUT_DIR" \
    --manifest "$MANIFEST" \
    --skip_existing True \
    2>&1 | tee "$BATCH_LOG"
set +x

echo "[done] all outputs under: $OUTPUT_DIR"
