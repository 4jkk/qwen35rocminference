# Benchmark methodology

## What is measured

The benchmark wrappers execute the existing llama.cpp CLI sequentially and extract its prompt-evaluation and token-generation timing records. Each repetition starts a fresh process and loads the same model. Model loading is not included in the two reported throughput measurements. The default llama.cpp warmup remains enabled.

Prompt processing evaluates the input tokens. Generation repeatedly predicts subsequent tokens while using accumulated state. They are different workloads: an improvement in prompt processing does not establish an improvement in generation.

Rates are the `tokens per second` values printed by the pinned llama.cpp revision. Reports use the arithmetic mean of those per-run rates, already rounded by llama.cpp. They do not derive rates from the configured maximum output length: llama.cpp's accounting, actual generated length, and end-of-sequence stopping can differ from that limit.

## Historical baselines, 2026-10-25

| Parameter | GPU baseline | CPU baseline |
| --- | --- | --- |
| Model | Qwen3.5-4B Q4_K_M | Same |
| llama.cpp revision | `df9beac7153166550571c70cc667f270b745da49` | Same local source pin |
| Executable | `misc/llama.cpp/build-gfx1032-test/bin/llama-cli` | Same script default |
| Context size | 512 | 512 |
| Maximum generated tokens | 128 | 128 |
| Logical batch (`-b`) | 32 | 32 |
| Physical microbatch (`-ub`) | 1 | 1 |
| GPU layers requested (`-ngl`) | 99 | 0 |
| GPU layers logged | 33/33 | 0/33 |
| Device argument | `--device ROCm0` | No explicit device argument |
| CPU threads | llama.cpp default; logs resolve to 6 | Explicit `-t 6` |
| Flash Attention | Off | Off |
| Reasoning | Off | Off |
| Repetitions | 3 | 3 |
| Sampling and seed | llama.cpp defaults; no explicit seed | Same |

The fixed user prompt is:

```text
Explain how GPU inference works and why memory bandwidth matters.
```

The model's chat template adds tokens around that text; saved logs report 24 prompt tokens. Both runs use `--single-turn`, and all six historical trials report the 128-token output limit.

Evidence is preserved in [GPU metadata](../results/gpu-baseline-20261025-123928/metadata.txt), [GPU summary](../results/gpu-baseline-20261025-123928/summary.txt), [CPU metadata](../results/cpu-baseline-20261025-163228/metadata.txt), and [CPU summary](../results/cpu-baseline-20261025-163228/summary.txt). Historical metadata is less complete than the new runner's format; it has not been retroactively edited. Configuration not recorded there is identified above from the scripts and raw logs.

`-ngl 99` requests offloading; the logs, not the request, establish that all 33 layers were offloaded. Conversely, `-ngl 0` records zero model layers offloaded but does not disable the GPU backend or prove that every operation/allocation was on the CPU. An explicit `--device none` CPU experiment would be a separately labeled configuration, not a silent replacement of this baseline.

## Saved measurements

| Backend / metric | Run 1 | Run 2 | Run 3 | Arithmetic mean |
| --- | ---: | ---: | ---: | ---: |
| GPU prompt | 5.82 | 5.83 | 5.83 | 5.827 |
| GPU generation | 4.83 | 4.84 | 4.84 | 4.837 |
| CPU prompt | 14.72 | 14.74 | 14.75 | 14.737 |
| CPU generation | 12.32 | 12.37 | 12.31 | 12.333 |

All rates are tokens/sec. These three-run observations describe this configuration, not statistical confidence or a general hardware ranking. The GPU slowdown's cause remains unproven.

## Commands and supported settings

Run `./scripts/benchmark-gpu.sh` or `./scripts/benchmark-cpu.sh`. Both delegate to `scripts/lib/benchmark.sh`. Defaults resolve relative to the repository, even when the wrapper is launched from elsewhere. Relative path overrides also resolve relative to the repository.

| Environment variable | Default | Applies to |
| --- | --- | --- |
| `LLAMA_CLI` | `misc/llama.cpp/build-gfx1032-test/bin/llama-cli` | Both |
| `MODEL` | `models/Qwen3.5-4B.Q4_K_M.gguf` | Both |
| `RESULTS_DIR` | `results` | Both |
| `RUNS` | `3` | Both |
| `CONTEXT_SIZE` | `512` | Both |
| `PREDICT_TOKENS` | `128` | Both; maximum, not guaranteed generated length |
| `BATCH_SIZE` | `32` | Both |
| `MICROBATCH_SIZE` | `1` | Both |
| `CPU_THREADS` | `6` | CPU only; GPU uses llama.cpp defaults |
| `DEVICE` | `ROCm0` | GPU only |
| `FLASH_ATTENTION` | `off` | GPU only; CPU is fixed to off |

Numeric values must be positive integers and microbatch must not exceed logical batch. GPU Flash Attention accepts `off`, `on`, or `auto`; only `off` is established by these baselines. Changing accepted options does not guarantee backend support. GPU device `none` is rejected. Missing executables and empty/missing models fail before a run directory is created.

The prompt, requested layer counts, reasoning mode, and single-turn behavior are fixed in the runner. Example configuration files are trusted shell snippets, loaded only when explicitly sourced and exported. The root `.env` is not auto-loaded. Avoid unrelated inherited llama.cpp/ROCm environment overrides when comparing experiments; they are not comprehensively captured by the runner.

## Result layout and failure handling

New directories have names such as `gpu-baseline-YYYYMMDD-HHMMSS-XXXXXX`; a unique suffix prevents collisions. The `baseline` prefix is retained for script compatibility; an overridden run must be interpreted using its metadata, not its directory name alone.

| File | Contents |
| --- | --- |
| `metadata.txt` | Date, executable/hash, committed pin, local source HEAD when available, settings, system/kernel, fixed prompt, sampling policy, and shell-escaped commands |
| `run-N.log` | llama.cpp's detailed logging output |
| `run-N.out` | Captured stdout/stderr, including generated text and errors |
| `summary.txt` | Layer-offload and prompt/generation timings for completed trials |
| `status.txt` | Starts as running; on exit records final status, exit code, and completed repetition count |

The local source HEAD is explicitly labeled as source state, not proof of how the binary was built. The CLI can be a small wrapper loading shared libraries, so its hash alone does not identify the complete runtime. Preserve compiler flags, backend library, and linked system-library versions when attributing performance changes.

A nonzero inference exit stops the benchmark and retains its logs. Missing timing pairs or expected offload evidence also fail the run. GPU validation requires positive layer offloading; inspect the actual count for the expected 33/33 result. CPU validation requires a logged zero-layer count. This runner does not independently monitor GPU utilization or validate numerical output correctness.

The EXIT/signal handlers retain failure status for ordinary exits and handled interrupts. A forced kill or machine failure may leave a running status or prevent the file from being written; neither establishes success. Historical runs predate this status format and are documented separately.

## Plotting and comparisons

`benchmarks/plot_results.py` accepts explicit CPU/GPU summary paths and a new output path. It rejects malformed timings, duplicate runs/metrics, missing timing pairs, nonpositive rates, and an explicitly failed status. It computes arithmetic means from the input records, labels CPU/GPU plots correctly, and never overwrites a chart. Historical summaries without status files remain readable.

Plotting alone does not verify comparable metadata. Confirm model, build, prompt, context, batch settings, output limit, sampling, threads, and placement before interpreting differences. Preserve raw evidence even when `.log` and `.out` happen to be byte-identical. Historical CPU chart labeling is known to be incorrect; the CPU metadata and text summaries identify the backend correctly.

## Limits and next experiment

The baselines lack a fixed sampling seed, full runtime provenance, controlled background-load measurements, and recorded power/thermal conditions. File/page caches and the enabled warmup may affect observations. Three repetitions are a small sample. Neither throughput nor successful text output establishes numerical model correctness.

The earlier `gfx1030` kernel-lookup crash and rocBLAS/Tensile abort are documented in [LOG.md](../LOG.md); they do not establish the cause of the working build's poor throughput. Increasing microbatch to 32 may reach a different kernel path, including the previously failing rocBLAS path, as well as require more memory.

The [next-experiment protocol](../experiments/README.md) changes only physical microbatch size. It has not been executed as part of the repository refactor.
