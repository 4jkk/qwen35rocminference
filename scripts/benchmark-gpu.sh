#!/usr/bin/env bash
set -Eeuo pipefail

# GPU inference benchmark for llama.cpp.
# Run from any directory; paths are resolved relative to this script.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_CLI="${LLAMA_CLI:-$ROOT_DIR/misc/llama.cpp/build-gfx1032-test/bin/llama-cli}"
MODEL="${MODEL:-$ROOT_DIR/models/Qwen3.5-4B.Q4_K_M.gguf}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
DEVICE="${DEVICE:-ROCm0}"

RUNS="${RUNS:-3}"
CONTEXT_SIZE="${CONTEXT_SIZE:-512}"
PREDICT_TOKENS="${PREDICT_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"
FLASH_ATTENTION="${FLASH_ATTENTION:-off}"

PROMPT="Explain how GPU inference works and why memory bandwidth matters."

# Validate inputs before starting.
if [[ ! -x "$LLAMA_CLI" ]]; then
    echo "ERROR: llama.cpp CLI not found or not executable: $LLAMA_CLI" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: RUNS must be a positive integer." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULTS_DIR/gpu-baseline-$TIMESTAMP"
mkdir -p "$RUN_DIR"

{
    echo "Benchmark: GPU inference"
    echo "Date: $(date --iso-8601=seconds)"
    echo "llama.cpp CLI: $LLAMA_CLI"
    echo "Model: $MODEL"
    echo "Device: $DEVICE"
    echo "Runs: $RUNS"
    echo "Context size: $CONTEXT_SIZE"
    echo "Prediction tokens: $PREDICT_TOKENS"
    echo "Batch size: $BATCH_SIZE"
    echo "Microbatch size: $MICROBATCH_SIZE"
    echo "Flash Attention: $FLASH_ATTENTION"
    echo "Prompt: $PROMPT"
    echo
} | tee "$RUN_DIR/metadata.txt"

for ((run = 1; run <= RUNS; run++)); do
    echo "========== GPU RUN $run/$RUNS =========="

    "$LLAMA_CLI" \
        --device "$DEVICE" \
        -m "$MODEL" \
        -ngl 99 \
        -b "$BATCH_SIZE" \
        -ub "$MICROBATCH_SIZE" \
        -c "$CONTEXT_SIZE" \
        -n "$PREDICT_TOKENS" \
        -fa "$FLASH_ATTENTION" \
        --reasoning off \
        --single-turn \
        --verbosity 4 \
        --log-file "$RUN_DIR/run-${run}.log" \
        -p "$PROMPT" \
        > "$RUN_DIR/run-${run}.out" 2>&1

    # Keep the relevant metrics in the terminal for quick inspection.
    grep -E 'offloaded .* layers|prompt eval time|eval time' \
        "$RUN_DIR/run-${run}.log" || true

    echo
done

# Create a compact summary from the diagnostic logs.
SUMMARY="$RUN_DIR/summary.txt"

{
    echo "GPU benchmark summary"
    echo "Results directory: $RUN_DIR"
    echo

    for ((run = 1; run <= RUNS; run++)); do
        echo "Run $run"
        grep -E 'offloaded .* layers|prompt eval time|eval time' \
            "$RUN_DIR/run-${run}.log" || true
        echo
    done
} > "$SUMMARY"

echo "Benchmark complete."
echo "Detailed results: $RUN_DIR"
echo "Summary: $SUMMARY"