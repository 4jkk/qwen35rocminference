#!/usr/bin/env bash
set -Eeuo pipefail

BACKEND="$1"
ROOT_DIR="$2"
cd "$ROOT_DIR"
[[ "$BACKEND" == cpu || "$BACKEND" == gpu ]] || { echo 'Expected cpu or gpu.' >&2; exit 1; }

LLAMA_CLI="${LLAMA_CLI:-$ROOT_DIR/misc/llama.cpp/build-gfx1032-test/bin/llama-cli}"
MODEL="${MODEL:-$ROOT_DIR/models/Qwen3.5-4B.Q4_K_M.gguf}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
RUNS="${RUNS:-3}"
CONTEXT_SIZE="${CONTEXT_SIZE:-512}"
PREDICT_TOKENS="${PREDICT_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"
CPU_THREADS="${CPU_THREADS:-6}"
DEVICE="${DEVICE:-ROCm0}"
FLASH_ATTENTION="${FLASH_ATTENTION:-off}"
PROMPT='Explain how GPU inference works and why memory bandwidth matters.'

for name in RUNS CONTEXT_SIZE PREDICT_TOKENS BATCH_SIZE MICROBATCH_SIZE; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer." >&2; exit 1; }
done
(( MICROBATCH_SIZE <= BATCH_SIZE )) || { echo 'MICROBATCH_SIZE must not exceed BATCH_SIZE.' >&2; exit 1; }
if [[ "$BACKEND" == cpu ]]; then
    [[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'CPU_THREADS must be a positive integer.' >&2; exit 1; }
    FLASH_ATTENTION=off
    placement=(-ngl 0 -t "$CPU_THREADS")
    device_description='backend default (no --device argument)'
    threads_description="$CPU_THREADS"
    layers=0
else
    [[ -n "$DEVICE" && "$DEVICE" != none ]] || { echo 'DEVICE must select a GPU.' >&2; exit 1; }
    [[ "$FLASH_ATTENTION" =~ ^(off|on|auto)$ ]] || { echo 'FLASH_ATTENTION must be off, on, or auto.' >&2; exit 1; }
    placement=(--device "$DEVICE" -ngl 99)
    device_description="$DEVICE"
    threads_description='llama.cpp default; see run logs for resolved thread counts'
    layers=99
fi
[[ -x "$LLAMA_CLI" ]] || { echo "CLI not executable: $LLAMA_CLI" >&2; exit 1; }
[[ -s "$MODEL" ]] || { echo "Model missing or empty: $MODEL" >&2; exit 1; }

mkdir -p "$RESULTS_DIR"
RUN_DIR="$(mktemp -d "$RESULTS_DIR/$BACKEND-baseline-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
completed=0
finish() {
    code=$?
    if (( code == 0 )); then state=completed; else state=failed; fi
    printf 'Status: %s\nExit code: %s\nCompleted runs: %s/%s\n' "$state" "$code" "$completed" "$RUNS" > "$RUN_DIR/status.txt"
    printf 'Results retained: %s\n' "$RUN_DIR"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Status: running\nCompleted runs: 0/%s\n' "$RUNS" > "$RUN_DIR/status.txt"

args=("$LLAMA_CLI" -m "$MODEL" "${placement[@]}" -b "$BATCH_SIZE" -ub "$MICROBATCH_SIZE"
    -c "$CONTEXT_SIZE" -n "$PREDICT_TOKENS" -fa "$FLASH_ATTENTION"
    --reasoning off --single-turn --verbosity 4)
{
    printf 'Benchmark: %s inference\nDate: %s\n' "$BACKEND" "$(date --iso-8601=seconds)"
    printf 'llama.cpp CLI: %s\nExecutable SHA256: %s\n' "$LLAMA_CLI" "$(sha256sum "$LLAMA_CLI" | cut -d ' ' -f 1)"
    printf 'Pinned llama.cpp commit: %s\n' "$(cat "$ROOT_DIR/configs/llama.cpp.commit")"
    if git -C "$ROOT_DIR/misc/llama.cpp" rev-parse HEAD >/dev/null 2>&1; then
        printf 'Local source HEAD (not binary provenance): %s\n' "$(git -C "$ROOT_DIR/misc/llama.cpp" rev-parse HEAD)"
    fi
    printf 'Model: %s\nDevice: %s\nCPU threads: %s\n' "$MODEL" "$device_description" "$threads_description"
    printf 'Runs: %s\nContext size: %s\nPrediction tokens: %s\n' "$RUNS" "$CONTEXT_SIZE" "$PREDICT_TOKENS"
    printf 'Batch size: %s\nMicrobatch size: %s\nGPU layers requested: %s\n' "$BATCH_SIZE" "$MICROBATCH_SIZE" "$layers"
    printf 'Flash Attention: %s\nReasoning: off\nSampling: llama.cpp defaults; no explicit seed\nPrompt: %s\n' "$FLASH_ATTENTION" "$PROMPT"
    printf 'System: %s\n' "$(uname -srmo)"
} > "$RUN_DIR/metadata.txt"
printf '%s benchmark summary\nResults directory: %s\n\n' "$BACKEND" "$RUN_DIR" > "$RUN_DIR/summary.txt"

for (( run=1; run<=RUNS; run++ )); do
    command=("${args[@]}" --log-file "$RUN_DIR/run-$run.log" -p "$PROMPT")
    { printf 'Run %s command:' "$run"; printf ' %q' "${command[@]}"; printf '\n'; } >> "$RUN_DIR/metadata.txt"
    printf 'Running %s trial %s/%s\n' "$BACKEND" "$run" "$RUNS"
    if "${command[@]}" > "$RUN_DIR/run-$run.out" 2>&1; then
        :
    else
        code=$?
        printf 'Trial %s failed with exit code %s; inspect %s\n' "$run" "$code" "$RUN_DIR" >&2
        exit "$code"
    fi
    log="$RUN_DIR/run-$run.log"
    if [[ ! -f "$log" ]] || ! grep -q 'prompt eval time' "$log" || ! grep -Eq '\|[[:space:]]+eval time' "$log"; then
        echo 'Trial has no complete prompt/generation timing pair.' >&2
        exit 1
    fi
    if [[ "$BACKEND" == gpu ]]; then
        grep -Eq 'offloaded [1-9][0-9]*/[1-9][0-9]* layers to GPU' "$log" || {
            echo 'Trial did not confirm GPU layer offloading.' >&2; exit 1;
        }
    else
        grep -Eq 'offloaded 0/[1-9][0-9]* layers to GPU' "$log" || {
            echo 'Trial did not confirm zero GPU layer offloading.' >&2; exit 1;
        }
    fi
    {
        printf 'Run %s\n' "$run"
        grep -E 'offloaded .* layers|prompt eval time|\|[[:space:]]+eval time' "$log"
        printf '\n'
    } >> "$RUN_DIR/summary.txt"
    completed=$run
done
cat "$RUN_DIR/summary.txt"
