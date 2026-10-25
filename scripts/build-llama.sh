#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
LLAMA_DIR="$ROOT_DIR/misc/llama.cpp"
PIN_FILE="$ROOT_DIR/configs/llama.cpp.commit"
GPU_TARGET="${AMDGPU_TARGETS:-gfx1032}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_DIR/build-gfx1032-test}"
BUILD_JOBS="${BUILD_JOBS:-2}"

[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'BUILD_JOBS must be a positive integer.' >&2; exit 1; }
[[ -f "$PIN_FILE" ]] || { echo "Missing revision pin: $PIN_FILE" >&2; exit 1; }
PINNED_COMMIT="$(tr -d '[:space:]' < "$PIN_FILE")"
[[ "$PINNED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || { echo 'Invalid llama.cpp revision pin.' >&2; exit 1; }

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    [[ ! -e "$LLAMA_DIR" ]] || { echo "$LLAMA_DIR exists but is not a Git checkout." >&2; exit 1; }
    mkdir -p "$ROOT_DIR/misc"
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

[[ -z "$(git -C "$LLAMA_DIR" status --porcelain)" ]] || {
    echo 'llama.cpp has local source changes; preserve them before building.' >&2
    exit 1
}
if [[ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
    git -C "$LLAMA_DIR" checkout --detach "$PINNED_COMMIT"
fi

args=(
    -S "$LLAMA_DIR" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_HIP=ON
    "-DCMAKE_HIP_ARCHITECTURES=$GPU_TARGET"
    -DGGML_HIP_NO_VMM=ON
    -DGGML_HIP_GRAPHS=ON
    -DGGML_HIP_MMQ_MFMA=ON
)
if [[ -n "${HIPCXX:-}" ]]; then
    args+=("-DCMAKE_HIP_COMPILER=$HIPCXX")
fi

printf 'Pinned revision: %s\nGPU target: %s\nBuild directory: %s\n' "$PINNED_COMMIT" "$GPU_TARGET" "$BUILD_DIR"
cmake "${args[@]}"
cmake --build "$BUILD_DIR" --target llama-cli llama-server --parallel "$BUILD_JOBS"
printf 'Inference CLI: %s/bin/llama-cli\n' "$BUILD_DIR"
