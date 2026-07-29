from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = OUT_DIR / "ltc_neuron_sweep_5ch_3ch_summary.csv"
OUT_PNG = OUT_DIR / "ltc_neuron_sweep_5ch_3ch_line.png"


def main() -> None:
    combined = pd.read_csv(SUMMARY_CSV).sort_values("Units")
    labels = combined["Architecture"].tolist()
    x = range(len(labels))

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
    ax.set_xlim(-0.18, len(labels) - 0.82)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.22)
    ax.legend(frameon=False, loc="lower right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
