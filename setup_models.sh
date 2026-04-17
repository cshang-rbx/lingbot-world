#!/usr/bin/env bash
#
# setup_models.sh -- ensure LingBot-World weights are present locally.
#
# Layout produced under <WEIGHT_DIR>:
#   <WEIGHT_DIR>/
#     Wan2.1_VAE.pth
#     models_t5_umt5-xxl-enc-bf16.pth
#     google/umt5-xxl/...
#     configuration.json
#     lingbot_world_fast/                 <-- robbyant/lingbot-world-fast
#       config.json
#       model-000XX-of-00016.safetensors  (16 shards)
#       diffusion_pytorch_model.safetensors.index.json
#
# The base-cam transformers (high_noise_model/, low_noise_model/) are skipped
# by default since the Fast inference pipeline uses lingbot_world_fast/ instead.
# Set INCLUDE_TRANSFORMERS=1 to pull them as well.
#
# Usage:
#   bash setup_models.sh <weight_dir>
#   INCLUDE_TRANSFORMERS=1 bash setup_models.sh <weight_dir>
#
# Tunables (env vars):
#   HF_REPO_BASE          repo id for base-cam  (default: robbyant/lingbot-world-base-cam)
#   HF_REPO_FAST          repo id for fast      (default: robbyant/lingbot-world-fast)
#   INCLUDE_TRANSFORMERS  1 to also download high_noise_model/ and low_noise_model/
#   HF_HUB_ENABLE_HF_TRANSFER  auto-enabled when hf_transfer is importable

set -eo pipefail

WEIGHT_DIR=${1:?weight_dir required (e.g. lingbot-world-base-cam)}
HF_REPO_BASE=${HF_REPO_BASE:-robbyant/lingbot-world-base-cam}
HF_REPO_FAST=${HF_REPO_FAST:-robbyant/lingbot-world-fast}
FAST_DIR="${WEIGHT_DIR}/lingbot_world_fast"

# Enable the Rust-based hf_transfer downloader when it is importable by the
# Python that backs huggingface-cli. This gives a significant speed-up on
# high-bandwidth links; we fall back silently otherwise.
_HF_CLI_PY=""
if command -v huggingface-cli >/dev/null 2>&1; then
    _HF_CLI_SHEBANG=$(head -n 1 "$(command -v huggingface-cli)" | sed 's|^#!||')
    # shellcheck disable=SC2206
    _HF_CLI_PY_CMD=($_HF_CLI_SHEBANG)
    _HF_CLI_PY="${_HF_CLI_PY_CMD[0]}"
fi
if [ -n "$_HF_CLI_PY" ] && [ -x "$_HF_CLI_PY" ] \
   && "$_HF_CLI_PY" -c 'import hf_transfer' >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "[setup] hf_transfer detected in $_HF_CLI_PY -- enabling fast transfer"
fi

# --- helpers --------------------------------------------------------------

# _need_any <path...>: return 0 (true, "need download") if ANY path is missing.
_need_any() {
    local f
    for f in "$@"; do
        if [ ! -e "$f" ]; then
            return 0
        fi
    done
    return 1
}

# _dl <repo_id> <local_dir> [extra hf-cli args...]
_dl() {
    local repo="$1"; shift
    local dst="$1"; shift
    mkdir -p "$dst"
    echo "[setup] huggingface-cli download $repo -> $dst"
    huggingface-cli download "$repo" --local-dir "$dst" "$@"
}

# --- 1) lingbot-world-base-cam (excluding transformers by default) -------

BASE_SENTINELS=(
    "${WEIGHT_DIR}/Wan2.1_VAE.pth"
    "${WEIGHT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
    "${WEIGHT_DIR}/configuration.json"
    "${WEIGHT_DIR}/google/umt5-xxl/spiece.model"
    "${WEIGHT_DIR}/google/umt5-xxl/tokenizer.json"
)

if _need_any "${BASE_SENTINELS[@]}"; then
    EXCLUDE_ARGS=()
    if [ "${INCLUDE_TRANSFORMERS:-0}" != "1" ]; then
        EXCLUDE_ARGS=(--exclude "high_noise_model/*" "low_noise_model/*")
        echo "[setup] ${HF_REPO_BASE}: excluding high_noise_model/ and low_noise_model/"
        echo "        (set INCLUDE_TRANSFORMERS=1 to include them)"
    else
        echo "[setup] ${HF_REPO_BASE}: INCLUDE_TRANSFORMERS=1 -- downloading full repo"
    fi
    _dl "$HF_REPO_BASE" "$WEIGHT_DIR" "${EXCLUDE_ARGS[@]}"
else
    echo "[setup] base-cam weights already present under ${WEIGHT_DIR} -- skipping"
fi

# --- 2) lingbot-world-fast -> <WEIGHT_DIR>/lingbot_world_fast -------------

# Sentinels: final shard + sharded index + config. If the first shard landed
# but the run got interrupted mid-download, huggingface-cli will resume.
FAST_SENTINELS=(
    "${FAST_DIR}/config.json"
    "${FAST_DIR}/diffusion_pytorch_model.safetensors.index.json"
    "${FAST_DIR}/model-00001-of-00016.safetensors"
    "${FAST_DIR}/model-00016-of-00016.safetensors"
)

if _need_any "${FAST_SENTINELS[@]}"; then
    _dl "$HF_REPO_FAST" "$FAST_DIR"
else
    echo "[setup] lingbot_world_fast weights already present under ${FAST_DIR} -- skipping"
fi

# --- 3) final sanity check -----------------------------------------------

MISSING=()
for f in "${BASE_SENTINELS[@]}" "${FAST_SENTINELS[@]}"; do
    [ -e "$f" ] || MISSING+=("$f")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "[setup][error] expected files still missing after download:" >&2
    for f in "${MISSING[@]}"; do echo "  - $f" >&2; done
    exit 1
fi

echo "[setup] OK: ${WEIGHT_DIR} is ready (fast dir: ${FAST_DIR})"
