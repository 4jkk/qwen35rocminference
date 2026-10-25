#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"

REPO_ID="${MODEL_REPO:-mradermacher/Qwen3.5-4B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-4B.Q4_K_M.gguf}"

MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

mkdir -p "$MODEL_DIR"

if [[ -s "$MODEL_PATH" ]]; then
    echo "Model already exists: $MODEL_PATH"
    exit 0
fi

if [[ -e "$MODEL_PATH" ]]; then
    echo "Error: model path exists but is empty: $MODEL_PATH" >&2
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "Error: Hugging Face CLI ('hf') is not installed." >&2
    echo "Install it with: uv tool install huggingface_hub" >&2
    exit 1
fi

echo "Downloading $MODEL_FILE from $REPO_ID..."
hf download "$REPO_ID" "$MODEL_FILE" \
    --local-dir "$MODEL_DIR"

if [[ ! -s "$MODEL_PATH" ]]; then
    echo "Error: download finished, but model file was not found." >&2
    exit 1
fi

echo "Model downloaded successfully:"
ls -lh "$MODEL_PATH"
