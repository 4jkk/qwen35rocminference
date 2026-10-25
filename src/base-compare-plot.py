import matplotlib.pyplot as plt
import numpy as np

# Average throughput measured in your three benchmark runs.
configurations = ["GPU (RX 6600)", "CPU (Ryzen 5 5600)"]

prompt_processing = [5.827, 14.737]
token_generation = [4.837, 12.333]

x = np.arange(len(configurations))
width = 0.34

fig, ax = plt.subplots(figsize=(9, 5.5))

prompt_bars = ax.bar(
    x - width / 2,
    prompt_processing,
    width,
    label="Prompt processing",
)

generation_bars = ax.bar(
    x + width / 2,
    token_generation,
    width,
    label="Token generation",
)

ax.set_title("Qwen3.5-4B: CPU vs GPU Inference Baseline")
ax.set_ylabel("Throughput (tokens/second)")
ax.set_xticks(x, configurations)
ax.grid(axis="y", alpha=0.25)
ax.legend()

ax.bar_label(prompt_bars, fmt="%.2f", padding=3)
ax.bar_label(generation_bars, fmt="%.2f", padding=3)

fig.tight_layout()
fig.savefig("results/baseline-comparison.png", dpi=160)
plt.show()

print(
    f"Prompt processing: CPU is {prompt_processing[1] / prompt_processing[0]:.2f}x faster"
)
print(
    f"Token generation: CPU is {token_generation[1] / token_generation[0]:.2f}x faster"
)
