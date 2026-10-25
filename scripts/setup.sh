
#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve the repository root from this script's location.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
GPU_TARGET="${AMDGPU_TARGETS:-gfx1030}"
echo "Project root: $ROOT_DIR"
echo "Build target: $GPU_TARGET"

echo "========================================"
echo " Qwen ROCm Inference — Environment Check"
echo "========================================"
echo "Project root: $ROOT_DIR"
echo

# Check required build and diagnostic tools.
missing=0

for cmd in git cmake c++ hipcc hipconfig rocminfo; do
    if command -v "$cmd" >/dev/null 2>&1; then
        printf '[OK]      %-12s %s\n' "$cmd" "$(command -v "$cmd")"
    else
        printf '[MISSING] %s\n' "$cmd"
        missing=1
    fi
done

echo
echo "Checking ROCm GPU detection..."

if command -v rocminfo >/dev/null 2>&1; then
    ROCM_INFO="$(rocminfo 2>&1 || true)"

    if [[ "$ROCM_INFO" == *"gfx1032"* ]]; then
        echo "[OK] ROCm detects gfx1032 (RX 6600)"
    else
        echo "[WARNING] gfx1032 not detected by rocminfo."
        echo "Check ROCm access before building llama.cpp."
    fi
fi

echo
if (( missing )); then
    echo "[ERROR] Required tools are missing."
    echo "Install missing prerequisites, then rerun this script."
    exit 1
fi

echo "[SUCCESS] Prerequisite checks passed."
 