# Qwen inference on CPU and AMD ROCm

This project investigates local LLM inference with llama.cpp, comparing CPU and AMD GPU performance before changing batch size, KV-cache settings, or Flash Attention. It currently provides shell-based inference benchmarks, saved baseline evidence, and Python plotting tools. The Python API client and metrics modules are placeholders.

**Current result:** ROCm/HIP inference completes with all 33 model layers offloaded. Under the recorded settings, GPU generation averages **4.84 tokens/sec**, while the zero-GPU-layer CPU baseline averages **12.33 tokens/sec**. These measurements do not establish why the GPU is slower.

## Tested configuration

| Component | Recorded configuration |
| --- | --- |
| OS | Fedora Linux 44 |
| CPU | AMD Ryzen 5 5600, 6 cores / 12 threads |
| GPU | AMD Radeon RX 6600, 8 GB VRAM; ROCm architecture `gfx1032` |
| ROCm | 7.1.1; HIP backend |
| llama.cpp | [`df9beac7153166550571c70cc667f270b745da49`](https://github.com/ggml-org/llama.cpp/tree/df9beac7153166550571c70cc667f270b745da49), 0.7.0-dev |
| Model | `models/Qwen3.5-4B.Q4_K_M.gguf`, Q4_K_M, 2,708,804,800 bytes |
| Working local executable | `misc/llama.cpp/build-gfx1032-test/bin/llama-cli` |
| Runtime settings | Context 512, logical batch 32, physical microbatch 1, Flash Attention off |

The revision is stored in [configs/llama.cpp.commit](configs/llama.cpp.commit). Models, upstream source, builds, and virtual environments are local files excluded from this repository. A fresh clone does not include them. This configuration is an observed working setup, not a claim of official support for this GPU.

## Run with the existing build

From the repository root:

```bash
./scripts/setup.sh

./misc/llama.cpp/build-gfx1032-test/bin/llama-cli \
  --device ROCm0 -m models/Qwen3.5-4B.Q4_K_M.gguf \
  -ngl 99 -c 512 -n 64 -b 32 -ub 1 -fa off \
  --reasoning off --single-turn --verbosity 4 -p "Hello"
```

`setup.sh` only checks installed tools, ROCm access, and the reported architecture. It installs nothing and does not prove kernel execution. For inference, require generated text, successful exit, and `offloaded 33/33 layers to GPU` in the logs. Device detection alone is insufficient.

Do not use the older local `misc/llama.cpp/build/bin/llama-cli` for these documented GPU commands: that build targets `gfx1030` and was associated with the earlier crashes. The working commands do not use `HSA_OVERRIDE_GFX_VERSION`.

## Prepare a fresh checkout

Prerequisites are Bash, Git, CMake, a C++ compiler, an installed HIP/ROCm development environment with working GPU device access, and `uv` for Python tooling. The scripts do not install or replace system ROCm packages.

```bash
uv sync --locked
./scripts/setup.sh
./scripts/download-model.sh
```

The download script uses the separately installed Hugging Face CLI (`hf`). It defaults to `mradermacher/Qwen3.5-4B-GGUF` and preserves an existing nonempty model. See [configs/model.env.example](configs/model.env.example) for repository/filename overrides.

Building is a separate action. It is unnecessary when the working executable already exists. To create a new build while preserving existing build directories:

```bash
BUILD_DIR="$PWD/misc/llama.cpp/build-gfx1032-new" \
  BUILD_JOBS=2 ./scripts/build-llama.sh
```

The build script clones upstream if absent, requires the committed revision pin, refuses a dirty source checkout, and builds `llama-cli` and `llama-server`. Defaults are `gfx1032`, Release, HIP, no VMM, HIP graphs enabled, and two compiler jobs. `HIPCXX` can select the HIP compiler explicitly. Without `BUILD_DIR`, it configures/builds **the existing `build-gfx1032-test` directory**, so use the override when preserving that artifact.

When using a different build, pass its executable through `LLAMA_CLI` to benchmarks. Source revision and wrapper-executable hash alone do not fully identify all linked libraries; keep the corresponding build configuration for comparisons.

## Collect measurements

These commands run real inference and create new result directories:

```bash
./scripts/benchmark-gpu.sh
./scripts/benchmark-cpu.sh
```

Both default to three runs, context 512, 128 maximum generated tokens, batch 32, and microbatch 1. GPU defaults to `ROCm0`, requests 99 GPU layers, and disables Flash Attention. CPU requests zero GPU layers, uses six threads, and disables Flash Attention. It retains the historical behavior of not passing `--device none`; it is a zero-layer-offload baseline, not verified complete GPU-device exclusion.

Settings are exported environment variables; example files are not automatically loaded:

```bash
set -a
source configs/benchmark.env.example
set +a
./scripts/benchmark-gpu.sh
```

See [benchmark methodology](docs/benchmark_methodology.md) for the complete variable table, historical settings, output format, validation rules, and limitations. New runs record settings, shell-escaped commands, exit status, timing summaries, and raw logs. Existing result directories are never reused. A failed run remains available for inspection.

## Results so far

Arithmetic means of the three per-run rates printed by llama.cpp:

| Baseline | Prompt processing | Token generation | Layer offloading |
| --- | ---: | ---: | --- |
| RX 6600 / ROCm | 5.827 tokens/sec | 4.837 tokens/sec | 33/33 |
| Ryzen 5 5600 / CPU baseline | 14.737 tokens/sec | 12.333 tokens/sec | 0/33 |

The CPU baseline is approximately 2.53 times faster for prompt processing and 2.55 times faster for generation in these runs. This is specific to this model, revision, and configuration. It is not a general comparison of the hardware.

- [GPU metadata](results/gpu-baseline-20261025-123928/metadata.txt) and [summary](results/gpu-baseline-20261025-123928/summary.txt)
- [CPU metadata](results/cpu-baseline-20261025-163228/metadata.txt) and [summary](results/cpu-baseline-20261025-163228/summary.txt)
- [Saved comparison chart](results/baseline-comparison.png)

Raw `.log` and `.out` files and historical PNGs are preserved unchanged. The historical CPU plot was generated by a script with an incorrect GPU title; consult metadata and summaries for the backend identity. The consolidated plotter generates the correct title without overwriting that historical image.

## Plot explicit result files

```bash
uv run --locked python benchmarks/plot_results.py gpu \
  --gpu-summary results/gpu-baseline-20261025-123928/summary.txt \
  --output /tmp/qwen-gpu-baseline-reviewed.png

uv run --locked python benchmarks/plot_results.py cpu \
  --cpu-summary results/cpu-baseline-20261025-163228/summary.txt \
  --output /tmp/qwen-cpu-baseline-reviewed.png

uv run --locked python benchmarks/plot_results.py compare \
  --cpu-summary results/cpu-baseline-20261025-163228/summary.txt \
  --gpu-summary results/gpu-baseline-20261025-123928/summary.txt \
  --output /tmp/qwen-baseline-comparison-reviewed.png
```

Output filenames must be new. PNG, PDF, and SVG are supported. Plotting is headless unless `--show` is supplied. Inputs are explicit: the tool never silently selects a different experiment. Verify that compared metadata describes compatible workloads. The previous plotting commands under `src/` have been replaced by these modes.

## Repository map and development

| Location | Responsibility |
| --- | --- |
| `scripts/` | Prerequisite checks, pinned build, model download, benchmark entry points; `scripts/lib/` holds shared benchmark execution |
| `configs/` | Exportable configuration examples and upstream revision pin |
| `benchmarks/plot_results.py` | Summary parsing and plots; `benchmarks/benchmarks.py` remains a placeholder |
| `src/qwen_inference/` | Python package scaffold; `client.py` and `metrics.py` are not implemented |
| `results/` | Versioned historical evidence and local outputs of future runs |
| `docs/`, `experiments/` | Architecture, methodology, and controlled experiment instructions |
| `tests/` | Tooling regression tests; `test_correctness.py` remains an empty model-correctness placeholder |
| `models/`, `misc/llama.cpp/`, `.venv/` | Ignored local assets |

```bash
uv run --locked pytest -q
bash -c 'for script in scripts/*.sh scripts/lib/*.sh; do bash -n "$script" || exit; done'
```

Tests use fake inference/build executables and temporary plot outputs; they do not build llama.cpp, load a model, or establish numerical model correctness.

For Git changes, inspect `git status --short`, stage explicit paths, and review `git diff --cached`. Do not stage the model, upstream checkout, builds, virtual environment, or local `.env` files. Existing baselines should remain immutable; write a new result directory for each experiment.

Further reading: [specification](SPEC.md), [architecture](docs/architecture.md), [methodology](docs/benchmark_methodology.md), [experiment protocol](experiments/README.md), and [project log](LOG.md).
