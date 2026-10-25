# Project log

## 2026-10-24: inference failure investigation

The reported Fedora/RX 6600 setup used ROCm 7.1.1, Qwen3.5-4B Q4_K_M, and llama.cpp pinned to `df9beac7153166550571c70cc667f270b745da49`. CPU inference worked. GPU runs failed even with Flash Attention disabled.

Read-only inspection found an RX 6600 reporting `gfx1032`, while the original `build/` HIP library contained `gfx1030` GPU bundles. Core PID 2076316 showed a SIGSEGV inside HIP kernel lookup/launch validation when invoking fused RMS normalization. An earlier core, PID 2053147, instead showed an abort during rocBLAS/Tensile library selection through `hipblasGemmBatchedEx`. Installed Tensile files lacked a `gfx1032` library.

These are two observed failure paths, not proof that one setting explains every crash. The core files and inspection transcripts are not committed in this repository. `HSA_OVERRIDE_GFX_VERSION` was not repeated, and system ROCm libraries were not replaced.

One diagnostic command unexpectedly extracted 141 host and 141 GPU bundle copies into the original build's `bin/` directory. They remain local generated artifacts and have not been deleted or committed.

## 2026-10-25: successful native HIP verification

An isolated build exported the unchanged pinned source into `/tmp/qwen-hip-gfx1032-77ruqnjy/` and compiled native `gfx1032` code. All 141 embedded GPU bundles in the resulting HIP library used that target.

The first test with logical batch 1 and physical microbatch 1 aborted in host batch setup while adding a second token. Increasing logical capacity to 32 while retaining physical microbatch 1 and Flash Attention off produced text successfully.

An eight-token test generated `Hello! How can I help you today` and exited successfully. A separate 64-token test completed with:

- `offloaded 33/33 layers to GPU`;
- logical batch 32 and physical microbatch 1;
- exit code 0;
- generation throughput 4.52 tokens/sec;
- 97-99% sampled GPU utilization during generation and driver registration of the inference process.

Its logs and telemetry remain in that temporary directory and may disappear after reboot or cleanup. They are not the later committed baseline runs. Native code plus the selected runtime settings is a confirmed working combination; avoidance of the problematic batched rocBLAS path remains an explanation to test.

## 2026-10-25: three-run baselines

The saved [GPU baseline](results/gpu-baseline-20261025-123928/summary.txt) used the repository-local `build-gfx1032-test` executable and reported 33/33 layers offloaded in all three runs. The [CPU baseline](results/cpu-baseline-20261025-163228/summary.txt) reported 0/33 layers offloaded. Both used context 512, 128 maximum generated tokens, batch 32, microbatch 1, and Flash Attention off.

| Arithmetic mean | GPU | CPU baseline |
| --- | ---: | ---: |
| Prompt processing | 5.827 tokens/sec | 14.737 tokens/sec |
| Token generation | 4.837 tokens/sec | 12.333 tokens/sec |

The GPU is slower in these measurements. The reason is not established. The CPU command does not explicitly exclude GPU devices, and neither baseline uses a fixed seed or comprehensive background-load/provenance capture. See [methodology](docs/benchmark_methodology.md).

## Repository cleanup

The `refactor/repository-docs` branch preserves existing tooling and baseline evidence before cleanup. The cleanup aligns native build defaults, shares benchmark execution, records future commands/status, consolidates plotting, removes redundant/no-op files, and fills documentation/configuration examples. Package/client/metrics/model-correctness scaffolds remain explicitly unfinished.

Historical metadata, logs, terminal captures, and PNGs are retained byte-for-byte. The historical CPU PNG may have a GPU title because its old plotting script used the wrong label; new plots correct the label without rewriting evidence. Historical logs retain their original whitespace.

This cleanup does not rebuild llama.cpp, change the model/source pin/system libraries, or run the proposed microbatch-32 experiment. Its validation uses temporary fixtures and fake executables.
