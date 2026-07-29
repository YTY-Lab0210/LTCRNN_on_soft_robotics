from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent

data = pd.DataFrame(
    {
        "mutation_rate": [0.001, 0.01, 0.1],
        "LTC-1": [55.83, 69.00, 71.83],
        "LTC-2": [63.67, 70.17, 72.00],
        "LTC-3": [67.17, 77.00, 65.83],
    }
)

csv_path = OUT_DIR / "pure_ga_fixed_mutation_rate_line.csv"
png_path = OUT_DIR / "pure_ga_fixed_mutation_rate_line.png"
data.to_csv(csv_path, index=False, encoding="utf-8-sig")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=220)
x = range(len(data))
colors = {
    "LTC-1": "#2878B5",
    "LTC-2": "#F28E2B",
    "LTC-3": "#59A14F",
}
markers = {"LTC-1": "o", "LTC-2": "s", "LTC-3": "^"}

for model in ["LTC-1", "LTC-2", "LTC-3"]:
    ax.plot(
        x,
        data[model],
        marker=markers[model],
        linewidth=2.8,
        markersize=8,
        color=colors[model],
        label=model,
    )
    label_offsets = {
        "LTC-1": [(0, 9), (0, -20), (0, -22)],
        "LTC-2": [(0, 9), (0, 13), (0, 12)],
        "LTC-3": [(0, 9), (0, 9), (0, 9)],
    }
    for idx, (xi, yi) in enumerate(zip(x, data[model])):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=label_offsets[model][idx],
            ha="center",
            fontsize=10,
            color=colors[model],
        )

ax.set_title("Pure-GA Fixed Mutation Rate Comparison", pad=14)
ax.set_xlabel("Mutation rate")
ax.set_ylabel("Mean test accuracy (%)")
ax.set_xticks(list(x), ["0.001", "0.01", "0.1"])
ax.set_ylim(50, 82)
ax.set_xlim(-0.1, 2.1)
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.25)
ax.legend(frameon=False, loc="upper left", ncols=3)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(png_path, bbox_inches="tight", facecolor="white")
print(png_path)
print(csv_path)
