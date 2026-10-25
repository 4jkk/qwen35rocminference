"""Plot explicit llama.cpp benchmark summaries without changing saved evidence."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Measurements:
    runs: tuple[int, ...]
    prompt: tuple[float, ...]
    generation: tuple[float, ...]

    @property
    def averages(self) -> tuple[float, float]:
        return sum(self.prompt) / len(self.runs), sum(self.generation) / len(self.runs)


def parse_summary(path: Path) -> Measurements:
    status_path = path.parent / "status.txt"
    if status_path.exists() and "Status: completed" not in status_path.read_text():
        raise ValueError(f"Benchmark is incomplete or failed: {status_path}")
    measurements: dict[int, dict[str, float]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = re.fullmatch(r"Run\s+([1-9][0-9]*)", line.strip())
        if header:
            current = int(header[1])
            if current in measurements:
                raise ValueError(f"Duplicate run {current}: {path}")
            measurements[current] = {}
            continue
        timing = line.rsplit("|", 1)[-1].strip()
        if timing.startswith("prompt eval time"):
            kind = "prompt"
        elif timing.startswith("eval time"):
            kind = "generation"
        else:
            continue
        match = re.search(r"(?:^|[\s,(])([0-9]+(?:\.[0-9]+)?)\s+tokens per second\)\s*$", timing)
        if current is None or match is None or float(match[1]) <= 0:
            raise ValueError(f"Invalid timing record: {line}")
        if kind in measurements[current]:
            raise ValueError(f"Duplicate {kind} timing for run {current}: {path}")
        measurements[current][kind] = float(match[1])
    if not measurements or any(set(values) != {"prompt", "generation"} for values in measurements.values()):
        raise ValueError(f"Expected one prompt and generation timing per run: {path}")
    runs = tuple(sorted(measurements))
    return Measurements(
        runs,
        tuple(measurements[run]["prompt"] for run in runs),
        tuple(measurements[run]["generation"] for run in runs),
    )


def make_figure(mode: str, cpu: Measurements | None, gpu: Measurements | None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if mode == "compare":
        assert cpu is not None and gpu is not None
        labels = ["CPU", "GPU"]
        prompt = [cpu.averages[0], gpu.averages[0]]
        generation = [cpu.averages[1], gpu.averages[1]]
        ax.set_title("Qwen3.5-4B: CPU vs GPU inference")
    else:
        values = cpu if mode == "cpu" else gpu
        assert values is not None
        labels = [f"Run {run}" for run in values.runs]
        prompt, generation = values.prompt, values.generation
        ax.set_title(f"Qwen3.5-4B {mode.upper()} baseline")
        ax.axhline(values.averages[0], linestyle="--", label=f"Prompt mean: {values.averages[0]:.2f}")
        ax.axhline(values.averages[1], linestyle=":", label=f"Generation mean: {values.averages[1]:.2f}")
    positions = list(range(len(labels)))
    prompt_bars = ax.bar([x - .17 for x in positions], prompt, .34, label="Prompt processing")
    generation_bars = ax.bar([x + .17 for x in positions], generation, .34, label="Token generation")
    ax.bar_label(prompt_bars, fmt="%.2f", padding=3)
    ax.bar_label(generation_bars, fmt="%.2f", padding=3)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Throughput (tokens/second)")
    ax.set_ylim(0, max(*prompt, *generation) * 1.25)
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cpu", "gpu", "compare"])
    parser.add_argument("--cpu-summary", type=Path)
    parser.add_argument("--gpu-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New .png, .pdf, or .svg file; never overwrite")
    parser.add_argument("--show", action="store_true", help="Also display an interactive window")
    args = parser.parse_args(argv)
    if args.mode in {"cpu", "compare"} and args.cpu_summary is None:
        parser.error("--cpu-summary is required for this mode")
    if args.mode in {"gpu", "compare"} and args.gpu_summary is None:
        parser.error("--gpu-summary is required for this mode")
    if args.output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--output must end in .png, .pdf, or .svg")
    if args.output.exists():
        parser.error(f"Output exists; choose a new filename: {args.output}")
    try:
        cpu = parse_summary(args.cpu_summary) if args.cpu_summary else None
        gpu = parse_summary(args.gpu_summary) if args.gpu_summary else None
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = make_figure(args.mode, cpu, gpu)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            fig.savefig(stream, format=args.output.suffix[1:].lower(), dpi=160, bbox_inches="tight")
        for name, values in [("CPU", cpu), ("GPU", gpu)]:
            if values is not None:
                print(f"{name}: prompt mean {values.averages[0]:.3f}, generation mean {values.averages[1]:.3f} tokens/sec")
        print(f"Chart: {args.output}")
        if args.show:
            plt.show()
        plt.close(fig)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
