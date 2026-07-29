from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


OUT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = OUT_DIR / "ltc_neuron_sweep_5ch_3ch_summary.csv"
OUT_PNG = OUT_DIR / "ltc_neuron_sweep_5ch_3ch_bar.png"
OUT_SVG = OUT_DIR / "ltc_neuron_sweep_5ch_3ch_bar.svg"


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV).sort_values("Units")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "legend.fontsize": 13,
            "hatch.linewidth": 1.2,
        }
    )

    labels = summary["Architecture"].tolist()
    x = np.arange(len(labels))
    width = 0.31

    color_3ch = "#efb2bd"
    color_5ch = "#c85a6a"
    edge = "#7d2035"

    fig, ax = plt.subplots(figsize=(5.45, 4.2), dpi=300)

    ax.bar(
        x - width / 2,
        summary["MeanAccuracy_3ch"],
        width=width,
        color=color_3ch,
        edgecolor=edge,
        linewidth=1.0,
        hatch="////",
        label="3 channel",
    )
    ax.bar(
        x + width / 2,
        summary["MeanAccuracy_5ch"],
        width=width,
        color=color_5ch,
        edgecolor=edge,
        linewidth=1.0,
        label="5 channel",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_xlabel("LTC model", fontsize=15, fontweight="bold")
    ax.set_ylabel("accuracy [%]", fontsize=15, fontweight="bold")
    ax.set_ylim(60, 100)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color("#111111")
    ax.tick_params(axis="both", width=1.3, length=5, color="#111111", labelsize=13)

    legend_handles = [
        Patch(facecolor=color_3ch, edgecolor=edge, hatch="////", label="3 channel"),
        Patch(facecolor=color_5ch, edgecolor=edge, label="5 channel"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=True,
        fancybox=False,
        edgecolor="#111111",
        framealpha=1.0,
        facecolor="white",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.24),
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.45)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
