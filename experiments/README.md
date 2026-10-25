# Controlled inference experiments

## Protect the reference

Keep both historical directories unchanged:

- [GPU baseline](../results/gpu-baseline-20261025-123928/metadata.txt)
- [CPU baseline](../results/cpu-baseline-20261025-163228/metadata.txt)

The native `build-gfx1032-test` executable and corresponding shared libraries are reference artifacts. Do not rebuild or delete them while attributing differences to a runtime setting. Do not change the GGUF or source pin between compared runs. New outputs belong in a new directory; they do not replace old logs, metadata, or charts.

## Next experiment: physical microbatch 32

**Status: proposed, not executed as part of the repository refactor.**

Question: does allowing larger physical microbatches improve prompt processing under the current HIP configuration, and what happens to generation throughput?

| Setting | Historical GPU reference | Proposed run |
| --- | --- | --- |
| Model and native build | Existing Qwen3.5-4B Q4_K_M / `gfx1032` | Same |
| Logical batch | 32 | 32 |
| Physical microbatch | 1 | 32 |
| Context | 512 | 512 |
| Maximum generated tokens | 128 | 128 |
| Flash Attention | Off | Off |
| GPU layers | 99 requested, 33/33 logged | Same request; verify logs |
| Repetitions | 3 | 3 |
| Prompt, sampling policy, threads | Historical defaults | Same |

From the repository root, with unrelated overrides removed and the historical default executable/model still selected:

```bash
git status --short
cat results/gpu-baseline-20261025-123928/metadata.txt
cat results/cpu-baseline-20261025-163228/metadata.txt

# This command starts the proposed experiment; it is not a setup step.
MICROBATCH_SIZE=32 ./scripts/benchmark-gpu.sh
```

The runner verifies and records the override, adds a unique directory suffix, and preserves failures. Review its printed result path and metadata rather than assuming any particular timestamp. Its `gpu-baseline-...` name is a historical naming convention, not a claim that an overridden configuration is the reference baseline.

Before comparison, check that `status.txt` says completed with 3/3 trials, each log reports the intended batch settings and 33/33 offloaded layers, generation completes, and both timing records are present. Compare prompt and generation rates separately and retain per-run values, not just a single mean.

Use `benchmarks/plot_results.py gpu --gpu-summary <new-directory>/summary.txt --output <new-chart>.png` to inspect the new runs. Calculate/reference the corresponding means in the historical [GPU summary](../results/gpu-baseline-20261025-123928/summary.txt). The plotter's `compare` mode is specifically CPU-versus-GPU; it is not a two-GPU-configuration experiment manager.

## If the run fails

A larger microbatch can consume more memory and select different HIP/rocBLAS kernels. The earlier investigation already observed a Tensile-selection failure, so do not assume that any failure is simply out of VRAM. Keep the failed directory and inspect `run-N.out`, `run-N.log` when created, and `status.txt` before selecting a remedy.

If evidence supports another size, try an intermediate physical microbatch such as 8 or 16 in a separately recorded invocation. Keep logical batch, Flash Attention, context, build, and model fixed. Do not combine a backend/library/source change with the batch experiment. Report a failure as a result rather than silently omitting it from a performance chart.

## Interpretation

This is a small, single-machine experiment. No fixed seed was used historically, and background activity/power conditions were not controlled. If stronger attribution is required, establish a new, separately labeled repeated reference under controlled conditions before comparing. Faster prompt processing alone does not prove faster autoregressive generation or explain the existing CPU/GPU gap.
