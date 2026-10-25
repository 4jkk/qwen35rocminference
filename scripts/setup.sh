#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TARGET="${AMDGPU_TARGETS:-gfx1032}"
printf 'Project: %s\nRequested GPU target: %s\n' "$ROOT_DIR" "$GPU_TARGET"

missing=0
for cmd in git cmake c++ hipcc hipconfig rocminfo; do
    if command -v "$cmd" >/dev/null 2>&1; then
        printf '[OK] %s: %s\n' "$cmd" "$(command -v "$cmd")"
    else
        printf '[MISSING] %s\n' "$cmd"
        missing=1
    fi
done
if (( missing )); then
    echo 'Prerequisite checks failed; no packages were installed.' >&2
    exit 1
fi

if ! info="$(rocminfo 2>&1)"; then
    printf '%s\n' "$info" >&2
    echo 'ROCm cannot inspect the GPU. Check device access before building.' >&2
    exit 1
fi
if [[ "$info" != *"$GPU_TARGET"* ]]; then
    echo "ROCm did not report $GPU_TARGET; check the architecture before building." >&2
    exit 1
fi
echo 'Prerequisite and architecture checks passed. Kernel execution has not been tested.'
