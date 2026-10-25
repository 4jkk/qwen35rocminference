# Architecture

## Execution path

```text
exported configuration
        |
scripts/benchmark-{gpu,cpu}.sh
        |
scripts/lib/benchmark.sh
        |
misc/llama.cpp/build-gfx1032-test/bin/llama-cli
        |
llama.cpp + ggml + CPU / HIP backend + system ROCm libraries
        |
models/Qwen3.5-4B.Q4_K_M.gguf
        |
results/<backend>-baseline-<timestamp>-<unique suffix>/
        |
benchmarks/plot_results.py -> a new chart
```

The model and inference engine remain upstream assets. Project scripts orchestrate processes and retain evidence; they do not implement tensor kernels or modify the GGUF. Python is used for plotting and tooling tests, not for the current inference loop.

## Boundaries

| Component | Responsibility | Outside its responsibility |
| --- | --- | --- |
| `scripts/setup.sh` | Check installed build tools, ROCm access, and reported target | Package installation; proof of successful inference |
| `scripts/build-llama.sh` | Clone/check pinned source and build a chosen output directory | Change the pin automatically; discard source edits; install ROCm |
| `scripts/download-model.sh` | Download the configured GGUF when absent | Replace a nonempty existing model; prove quantization quality |
| Benchmark wrappers/shared runner | Resolve settings, invoke CLI, retain status/logs, extract timings | Tune kernels, certify CPU-device isolation, monitor utilization |
| Plotting CLI | Parse explicit summaries and create labeled charts | Run inference, select a latest experiment, assert causality |
| `src/qwen_inference/` | Reserved Python package interface | Current inference orchestration; client/metrics are unfinished |

The `qwen-inference` Python entry point still prints a greeting. `client.py`, `metrics.py`, `benchmarks/benchmarks.py`, and `tests/test_correctness.py` are retained placeholders, not advertised features. Project tooling tests are separate from model-correctness tests.

## Source, builds, and system libraries

`configs/llama.cpp.commit` pins the upstream source to `df9beac7153166550571c70cc667f270b745da49`. The local checkout under `misc/llama.cpp/` has its own Git metadata and is ignored by the outer repository. The outer repository versions the pin and scripts, not upstream source or binaries.

Two local build directories existed before this refactor:

- `build/`: HIP code compiled for `gfx1030`, associated with the earlier kernel-lookup crash.
- `build-gfx1032-test/`: native `gfx1032`, used by the saved successful GPU benchmarks.

Both are preserved. Setup/build defaults now match the latter path and target, but merely changing a script does not modify or validate a preexisting binary. Rebuilding is explicit; use a fresh `BUILD_DIR` to preserve an artifact used in a baseline. The build script requires a clean source checkout and checks out the recorded revision only when needed. It never creates a new pin from whatever revision happens to be present.

The known build uses Release optimization, HIP, disabled VMM, enabled HIP graphs, and enabled HIP MMQ MFMA configuration. These settings reproduce the established configuration rather than demonstrating that each option is independently necessary. HIP matrix operations can use custom kernels or call hipBLAS/rocBLAS, which in turn selects Tensile kernels. The installed ROCm libraries are system dependencies and are not bundled or replaced by this repository.

The actual GPU reports `gfx1032`. A `gfx1030` compiler target is not equivalent to native-target coverage in the observed setup. The native build plus a one-token physical microbatch completed inference; the exact kernel-dispatch explanation for all prior failures has not been independently established. See [the investigation log](../LOG.md).

## Configuration and artifacts

Environment values are exported by the invoking shell. The two `.env.example` files document supported variables and are never automatically sourced. Paths supplied to benchmark/build scripts are resolved after changing to the project root. Download filenames/repository and inference-model paths are distinct settings.

The runner writes a unique result directory for each invocation and keeps each trial's command, log, terminal output, and timings. Completion/failure is separate from throughput values. Historical baseline directories have the older format and remain immutable; future metadata is richer without rewriting history.

Models, `.venv`, local `.env` settings, and `misc/` are excluded by `.gitignore`. Raw baseline results are deliberately versioned. Git ignores are safeguards against ordinary staging, not a substitute for reviewing explicit staged paths.

## Repository cleanup

The three duplicated/hard-coded `src/*-plot.py` utilities were replaced by `benchmarks/plot_results.py`. Removed files also include empty serving/cleanup scripts, the empty project-root CMake preset, the empty stray plotting module, and a heading-only result file. Existing baseline PNGs and duplicate raw captures were retained as historical evidence. The build script no longer creates an unused `third_party` directory.

No upstream code, model, system ROCm package, existing build, or baseline artifact is changed by this cleanup. New model-serving or Python-client work requires its own design and validation.
