import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT_DIR / "results"
SUMMARY_NAME = "summary.txt"


def find_latest_summary() -> Path:
    summaries = sorted(
        RESULTS_ROOT.glob("gpu-baseline-*/summary.txt"),
        key=lambda path: path.parent.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        raise FileNotFoundError(f"No GPU baseline summary found under {RESULTS_ROOT}")
    return summaries[0]


def parse_summary(summary_path: Path):
    text = summary_path.read_text(encoding="utf-8")
    prompt_values = {}
    generation_values = {}

    current_run = None
    for line in text.splitlines():
        run_match = re.match(r"Run\s+(\d+)", line.strip())
        if run_match:
            current_run = int(run_match.group(1))
            continue

        if current_run is None:
            continue

        prompt_match = re.search(
            r"prompt eval time\s*=.*?([0-9.]+)\s+tokens per second",
            line,
        )
        if prompt_match:
            prompt_values[current_run] = float(prompt_match.group(1))

        generation_match = re.search(
            r"\beval time\s*=.*?([0-9.]+)\s+tokens per second",
            line,
        )
        if generation_match:
            generation_values[current_run] = float(generation_match.group(1))

    runs = sorted(set(prompt_values) & set(generation_values))
    if not runs:
        raise ValueError(
            f"No prompt/generation throughput metrics found in {summary_path}"
        )

    return (
        runs,
        [prompt_values[run] for run in runs],
        [generation_values[run] for run in runs],
    )


def main():
    summary_path = find_latest_summary()
    runs, prompt_tps, generation_tps = parse_summary(summary_path)

    avg_prompt = sum(prompt_tps) / len(prompt_tps)
    avg_generation = sum(generation_tps) / len(generation_tps)

    x_positions = list(range(len(runs)))
    bar_width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.5))
    prompt_bars = ax.bar(
        [x - bar_width / 2 for x in x_positions],
        prompt_tps,
        bar_width,
        label="Prompt processing",
    )
    generation_bars = ax.bar(
        [x + bar_width / 2 for x in x_positions],
        generation_tps,
        bar_width,
        label="Token generation",
    )

    ax.axhline(
        avg_prompt,
        linestyle="--",
        linewidth=1.3,
        label=f"Prompt average: {avg_prompt:.2f} tok/s",
    )
    ax.axhline(
        avg_generation,
        linestyle=":",
        linewidth=1.5,
        label=f"Generation average: {avg_generation:.2f} tok/s",
    )

    ax.set_title("Qwen3.5-4B GPU Baseline — AMD Radeon RX 6600")
    ax.set_xlabel("Benchmark run")
    ax.set_ylabel("Throughput (tokens/second)")
    ax.set_xticks(x_positions, [f"Run {run}" for run in runs])
    ax.set_ylim(0, max(prompt_tps + generation_tps) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(prompt_bars, fmt="%.2f", padding=3)
    ax.bar_label(generation_bars, fmt="%.2f", padding=3)
    fig.tight_layout()

    output_path = summary_path.parent / "gpu-baseline-performance.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Summary: {summary_path}")
    print(f"Chart:   {output_path}")
    print(f"Average prompt processing: {avg_prompt:.3f} tokens/second")
    print(f"Average token generation:  {avg_generation:.3f} tokens/second")
    plt.show()


if __name__ == "__main__":
    main()
