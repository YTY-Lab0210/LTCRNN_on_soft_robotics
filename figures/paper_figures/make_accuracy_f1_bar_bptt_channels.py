from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(r"C:\Users\HAO\Desktop\YTY_from_macbook")
OUT_DIR = ROOT / "論文" / "paper_figures"
SOURCE_TABLE_DIR = OUT_DIR / "source_tables"

five_ch_summary = (
    SOURCE_TABLE_DIR
    / "ltc4_bptt_5ch_selected30_best_composite_summary.csv"
)
three_ch_summary = (
    SOURCE_TABLE_DIR
    / "ltc4_bptt_3ch_selected30_best_composite_summary.csv"
)


def read_bptt_metrics(path):
    df = pd.read_csv(path)
    acc = float(df.loc[df["metric"] == "Test Accuracy (%)", "bptt_mean"].iloc[0])
    f1 = float(df.loc[df["metric"] == "Macro-F1 (%)", "bptt_mean"].iloc[0])
    return acc, f1


acc_3ch, f1_3ch = read_bptt_metrics(three_ch_summary)
acc_5ch, f1_5ch = read_bptt_metrics(five_ch_summary)

data = pd.DataFrame(
    [
        {"Input": "3-channel", "Accuracy": acc_3ch, "Macro-F1": f1_3ch},
        {"Input": "5-channel", "Accuracy": acc_5ch, "Macro-F1": f1_5ch},
    ]
)

csv_path = OUT_DIR / "accuracy_f1_bptt_channels.csv"
png_path = OUT_DIR / "accuracy_f1_bptt_channels_bar.png"
data.to_csv(csv_path, index=False, encoding="utf-8-sig")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "hatch.linewidth": 1.5,
    }
)

fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=220)

groups = data["Input"].tolist()
x = range(len(groups))
bar_w = 0.22
gap = 0.16

accuracy_color = "#c85a6a"
f1_color = "#efb2bd"
edge_color = "#9b2f47"

bars_acc = ax.bar(
    [i - (bar_w + gap) / 2 for i in x],
    data["Accuracy"],
    width=bar_w,
    color=accuracy_color,
    edgecolor=edge_color,
    linewidth=1.0,
    label="Accuracy",
)
bars_f1 = ax.bar(
    [i + (bar_w + gap) / 2 for i in x],
    data["Macro-F1"],
    width=bar_w,
    color=f1_color,
    edgecolor=edge_color,
    linewidth=1.0,
    hatch="----",
    label="Macro-F1",
)

ax.set_title("LTC-4 BPTT Performance", pad=12)
ax.set_ylabel("Accuracy and F1-score (%)")
ax.set_xticks(list(x), groups)
ax.set_ylim(60, 100)
ax.set_yticks([60, 70, 80, 90, 100])
ax.set_xlim(-0.55, len(groups) - 0.45)
ax.grid(False)
ax.set_axisbelow(True)

legend_handles = [
    Patch(facecolor=accuracy_color, edgecolor=edge_color, label="Accuracy"),
    Patch(facecolor=f1_color, edgecolor=edge_color, hatch="----", label="F1-score"),
]
ax.legend(handles=legend_handles, frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)

fig.tight_layout()
fig.savefig(png_path, bbox_inches="tight", facecolor="white")
print(png_path)
print(csv_path)
