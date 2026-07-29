from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\HAO\Desktop\YTY_from_macbook")
OUT_DIR = ROOT / "論文" / "paper_figures"

five_ch_raw_path = ROOT / "LTCRNN" / "BPTT" / "csv" / "LTCRNN_Ablation_Raw_Data.csv"
three_ch_summary_path = ROOT / "LTCRNN" / "finger_3" / "BPTT" / "LTC_Neuron_Sweep_3ch_Summary.csv"

order = [1, 2, 4, 8, 16]
labels = [f"LTC-{n}" for n in order]

five_raw = pd.read_csv(five_ch_raw_path)
five_summary = (
    five_raw.assign(Units=five_raw["Architecture"].str.extract(r"LTC-(\d+)").astype(int))
    .groupby(["Architecture", "Units"], as_index=False)["Accuracy(%)"]
    .agg(MeanAccuracy="mean", StdAccuracy="std", MinAccuracy="min", MaxAccuracy="max", Runs="count")
)

three_summary = pd.read_csv(three_ch_summary_path)

rows = []
for units in order:
    arch = f"LTC-{units}"
    five_row = five_summary.loc[five_summary["Units"] == units].iloc[0]
    three_row = three_summary.loc[three_summary["Units"] == units].iloc[0]
    rows.append(
        {
            "Architecture": arch,
            "Units": units,
            "MeanAccuracy_5ch": five_row["MeanAccuracy"],
            "StdAccuracy_5ch": five_row["StdAccuracy"],
            "Runs_5ch": int(five_row["Runs"]),
            "MeanAccuracy_3ch": three_row["MeanAccuracy"],
            "StdAccuracy_3ch": three_row["StdAccuracy"],
            "Runs_3ch": int(three_row["Runs"]),
        }
    )

combined = pd.DataFrame(rows)
csv_path = OUT_DIR / "ltc_neuron_sweep_5ch_3ch_summary.csv"
png_path = OUT_DIR / "ltc_neuron_sweep_5ch_3ch_line.png"
combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

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
x = range(len(order))

series = [
    ("5 channels", combined["MeanAccuracy_5ch"], "#2878B5", "o"),
    ("3 channels", combined["MeanAccuracy_3ch"], "#F28E2B", "s"),
]

label_offsets = {
    "5 channels": [(-10, -22), (0, 12), (0, 12), (0, 12), (0, 12)],
    "3 channels": [(12, 10), (0, -20), (0, -20), (0, -20), (0, -20)],
}

for name, values, color, marker in series:
    ax.plot(
        x,
        values,
        marker=marker,
        markersize=8,
        linewidth=2.8,
        color=color,
        label=name,
    )
    for idx, (xi, yi) in enumerate(zip(x, values)):
        ax.annotate(
            f"{yi:.1f}",
            (xi, yi),
            textcoords="offset points",
            xytext=label_offsets[name][idx],
            ha="center",
            fontsize=10,
            color=color,
        )

ax.set_title("LTC Neuron Sweep: 5-Channel vs 3-Channel", pad=14)
ax.set_xlabel("LTC model")
ax.set_ylabel("Mean test accuracy (%)")
ax.set_xticks(list(x), labels)
ax.set_ylim(60, 96)
ax.set_xlim(-0.18, len(order) - 0.82)
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.22)
ax.legend(frameon=False, loc="lower right")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(png_path, bbox_inches="tight", facecolor="white")
print(png_path)
print(csv_path)
