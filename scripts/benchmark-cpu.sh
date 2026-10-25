#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_CLI="${LLAMA_CLI:-$ROOT_DIR/misc/llama.cpp/build-gfx1032-test/bin/llama-cli}"
MODEL="${MODEL:-$ROOT_DIR/models/Qwen3.5-4B.Q4_K_M.gguf}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"

RUNS="${RUNS:-3}"
CPU_THREADS="${CPU_THREADS:-6}"
CONTEXT_SIZE="${CONTEXT_SIZE:-512}"
PREDICT_TOKENS="${PREDICT_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"

PROMPT="Explain how GPU inference works and why memory bandwidth matters."

if [[ ! -x "$LLAMA_CLI" ]]; then
    echo "ERROR: llama.cpp CLI not found or not executable: $LLAMA_CLI" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found: $MODEL" >&2
    exit 1
fi

if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: RUNS must be a positive integer." >&2
    exit 1
fi

if [[ ! "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CPU_THREADS must be a positive integer." >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULTS_DIR/cpu-baseline-$TIMESTAMP"
mkdir -p "$RUN_DIR"

{
    echo "Benchmark: CPU inference"
    echo "Date: $(date --iso-8601=seconds)"
    echo "Model: $MODEL"
    echo "CPU threads: $CPU_THREADS"
    echo "Runs: $RUNS"
    echo "Context size: $CONTEXT_SIZE"
    echo "Prediction tokens: $PREDICT_TOKENS"
    echo "Batch size: $BATCH_SIZE"
    echo "Microbatch size: $MICROBATCH_SIZE"
    echo "GPU layers: 0"
    echo "Prompt: $PROMPT"
} | tee "$RUN_DIR/metadata.txt"

for ((run = 1; run <= RUNS; run++)); do
    echo
    echo "========== CPU RUN $run/$RUNS =========="

    "$LLAMA_CLI" \
        -m "$MODEL" \
        -ngl 0 \
        -t "$CPU_THREADS" \
        -b "$BATCH_SIZE" \
        -ub "$MICROBATCH_SIZE" \
        -c "$CONTEXT_SIZE" \
        -n "$PREDICT_TOKENS" \
        -fa off \
        --reasoning off \
        --single-turn \
        --verbosity 4 \
        --log-file "$RUN_DIR/run-${run}.log" \
        -p "$PROMPT" \
        > "$RUN_DIR/run-${run}.out" 2>&1

    grep -E 'offloaded .* layers|prompt eval time|eval time' \
        "$RUN_DIR/run-${run}.log" || true

    echo
done

SUMMARY="$RUN_DIR/summary.txt"

{
    echo "CPU benchmark summary"
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