import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLOT = ROOT / "benchmarks/plot_results.py"
spec = importlib.util.spec_from_file_location("plot_results", PLOT)
plots = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plots
spec.loader.exec_module(plots)
GPU = ROOT / "results/gpu-baseline-20261025-123928/summary.txt"
CPU = ROOT / "results/cpu-baseline-20261025-163228/summary.txt"


def test_historical_metrics_are_separate():
    gpu, cpu = plots.parse_summary(GPU), plots.parse_summary(CPU)
    assert gpu.runs == cpu.runs == (1, 2, 3)
    assert gpu.prompt == (5.82, 5.83, 5.83)
    assert gpu.generation == (4.83, 4.84, 4.84)
    assert cpu.prompt == (14.72, 14.74, 14.75)
    assert cpu.generation == (12.32, 12.37, 12.31)
    assert gpu.averages == pytest.approx((5.826666667, 4.836666667))
    assert cpu.averages == pytest.approx((14.736666667, 12.333333333))


@pytest.mark.parametrize("text", [
    "", "Run 1\n", "Run 1\nRun 1\n", "Run 1\n| eval time = malformed\n",
    "Run 1\n| prompt eval time = 1 ms (5.00 tokens per second)\n",
    "Run 1\n| eval time = 1 ms (0.00 tokens per second)\n",
    "Run 1\n| eval time = 1 ms (-2.00 tokens per second)\n",
    "Run 1\n| eval time = 1 ms (2.00 tokens per second)\n| eval time = 1 ms (3.00 tokens per second)\n",
])
def test_rejects_incomplete_or_malformed_summary(tmp_path, text):
    summary = tmp_path / "summary.txt"
    summary.write_text(text)
    with pytest.raises(ValueError):
        plots.parse_summary(summary)


@pytest.mark.parametrize("state", ["failed", "running"])
def test_rejects_failed_or_running_benchmark(tmp_path, state):
    summary = tmp_path / "summary.txt"
    summary.write_bytes(GPU.read_bytes())
    (tmp_path / "status.txt").write_text(f"Status: {state}\n")
    with pytest.raises(ValueError, match="incomplete or failed"):
        plots.parse_summary(summary)


def test_cpu_label_and_comparison_values():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cpu, gpu = plots.parse_summary(CPU), plots.parse_summary(GPU)
    figure = plots.make_figure("cpu", cpu, None)
    assert "CPU" in figure.axes[0].get_title()
    assert "GPU" not in figure.axes[0].get_title()
    plt.close(figure)
    figure = plots.make_figure("compare", cpu, gpu)
    assert [bar.get_height() for bar in figure.axes[0].patches] == pytest.approx([
        cpu.averages[0], gpu.averages[0], cpu.averages[1], gpu.averages[1],
    ])
    plt.close(figure)


@pytest.mark.parametrize("mode", ["cpu", "gpu", "compare"])
def test_headless_cli_and_output_protection(tmp_path, mode):
    output = tmp_path / f"{mode}.png"
    command = [sys.executable, str(PLOT), mode, "--output", str(output)]
    if mode in {"cpu", "compare"}:
        command += ["--cpu-summary", str(CPU)]
    if mode in {"gpu", "compare"}:
        command += ["--gpu-summary", str(GPU)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    assert original.startswith(b"\x89PNG")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert output.read_bytes() == original
