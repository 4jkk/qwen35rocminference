
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/misc/llama.cpp"
PIN_FILE="$ROOT_DIR/configs/llama.cpp.commit"

REPO_URL="https://github.com/ggml-org/llama.cpp.git"
GPU_TARGET="${AMDGPU_TARGETS:-gfx1030}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

mkdir -p "$ROOT_DIR/third_party" "$ROOT_DIR/configs"

# Clone the source if it does not exist.
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    if [[ -e "$LLAMA_DIR" ]]; then
        echo "Error: $LLAMA_DIR exists but is not a Git repository." >&2
        exit 1
    fi

    echo "Cloning llama.cpp..."
    git clone "$REPO_URL" "$LLAMA_DIR"
fi

# Pin the source revision on the first run.
if [[ -f "$PIN_FILE" ]]; then
    PINNED_COMMIT="$(tr -d '[:space:]' < "$PIN_FILE")"

    if [[ ! "$PINNED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "Error: invalid commit SHA in $PIN_FILE" >&2
        exit 1
    fi

    echo "Using pinned commit: $PINNED_COMMIT"
    git -C "$LLAMA_DIR" checkout --detach "$PINNED_COMMIT"
else
    RESOLVED_COMMIT="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
    printf '%s\n' "$RESOLVED_COMMIT" > "$PIN_FILE"

    echo "Pinned llama.cpp at: $RESOLVED_COMMIT"
    echo "Commit configs/llama.cpp.commit to preserve this revision."
fi

# Configure the HIP backend.
echo "Configuring llama.cpp for AMD GPU target: $GPU_TARGET"

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="$GPU_TARGET" \
    -DLLAMA_CURL=ON

# Compile the binaries.
echo "Building with $BUILD_JOBS parallel jobs..."

cmake --build "$LLAMA_DIR/build" \
    --config Release \
    --parallel "$BUILD_JOBS"

echo
echo "Build completed."

for binary in llama-cli llama-server; do
    if [[ -x "$LLAMA_DIR/build/bin/$binary" ]]; then
        echo "[OK] $LLAMA_DIR/build/bin/$binary"
    else
        echo "[WARNING] $binary was not found in build/bin."
    fi
done
