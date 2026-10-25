# Project specification

## Objective

Investigate and benchmark local LLM inference using llama.cpp on CPU and AMD ROCm/HIP, then evaluate individual changes such as batch size, KV-cache configuration, and Flash Attention using retained evidence.

## Current supported workflow

1. Check local prerequisites and the reported GPU architecture.
2. Use the existing native `gfx1032` build, or explicitly build the pinned source into a chosen directory.
3. Use the existing Qwen3.5-4B Q4_K_M GGUF, or explicitly download it when absent.
4. Run separate GPU and zero-GPU-layer CPU benchmarks with recorded settings.
5. Preserve raw output, timing summaries, and run status.
6. Plot explicit result summaries without overwriting evidence.

The tested environment and executable path are listed in [README.md](README.md). Other models, architectures, revisions, and backend settings are not automatically validated by accepting a path or configuration value.

## Interfaces

- Shell entry points: `scripts/setup.sh`, `scripts/build-llama.sh`, `scripts/download-model.sh`, `scripts/benchmark-gpu.sh`, and `scripts/benchmark-cpu.sh`.
- Configuration: exported environment variables described in [the methodology](docs/benchmark_methodology.md) and the two files under `configs/`. The upstream pin is a committed 40-character Git SHA.
- Plotting: `python benchmarks/plot_results.py {cpu,gpu,compare}` with explicit summary paths and required `--output`; `--show` is optional. See `--help` for syntax.
- Results: one unique directory per invocation, containing metadata, per-run logs/output, a timing summary, and final status where the process can write it.

Scripts must honor their documented values and record the command they actually invoke. The executable's initialization log is the evidence for resolved context/batch settings and actual layer placement; requested settings alone are insufficient.

## Acceptance criteria

For a real GPU validation, text generation must finish without a crash, logs must confirm the expected layer offloading, and separate device telemetry must establish GPU activity. The earlier investigation met this with a 64-token run, 33/33 layers, exit code 0, and 97-99% generation-period GPU utilization. That telemetry resides in a temporary diagnostic directory, not in the committed baseline format. Saved baseline logs independently establish repeated generation and offloading.

For a repository tooling change:

- Existing baseline bytes, model, upstream revision/source, and existing builds remain unchanged.
- Mocked process tests verify settings, metadata, placement checks, failure retention, and collision-free directories without requiring a GPU.
- Summary parsing rejects malformed/incomplete records and separates prompt/generation rates.
- Plot labels and values match the explicit inputs; existing output files cannot be overwritten.
- Documentation describes measured behavior, limitations, and untested work accurately.
- Staged changes exclude models, local environments, source checkouts, and compiled builds.

Tooling tests do not substitute for real inference validation after a future runtime/build change.

## Defaults and non-goals

Baseline defaults remain context 512, 128 maximum generated tokens, logical batch 32, physical microbatch 1, Flash Attention off, reasoning off, three repetitions, and the fixed documented prompt. GPU requests 99 layers on `ROCm0`; CPU requests zero layers and six threads without an explicit device argument.

The current project does not provide an implemented Python inference client, metrics API, model-correctness suite, serving wrapper, automatic tuning, or numerical equivalence guarantees. The greeting-only package entry point remains a scaffold. No Vulkan/CUDA fallback, architecture override, system ROCm replacement, automatic pin update, or inference optimization is part of this refactor.
