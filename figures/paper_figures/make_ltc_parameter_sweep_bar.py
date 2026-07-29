from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
UNITS = [1, 2, 4, 8, 16]
NUM_CLASSES = 10


def ltc_params(input_channels: int, units: int) -> int:
    ltc_layer = 3 * input_channels * units  # w, gamma/r, and mu
    classifier = units * NUM_CLASSES + NUM_CLASSES
    return ltc_layer + classifier


DATA = pd.DataFrame(
    {
        "Model": [f"LTC-{units}" for units in UNITS],
        "Units": UNITS,
        "5-channel": [ltc_params(5, units) for units in UNITS],
        "3-channel": [ltc_params(3, units) for units in UNITS],
    }
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA.to_csv(OUT_DIR / "ltc_parameter_sweep_counts.csv", index=False)

    models = DATA["Model"].tolist()
    x = np.arange(len(models))
    width = 0.34

    color_5ch = "#4C78A8"
    color_3ch = "#F58518"
    edge = "#303030"

    fig, ax = plt.subplots(figsize=(9.4, 4.9), dpi=300)

    bars_5 = ax.bar(
        x - width / 2,
        DATA["5-channel"],
        width,
        label="5-channel",
        color=color_5ch,
        edgecolor=edge,
        linewidth=0.8,
        alpha=0.82,
    )
    bars_3 = ax.bar(
        x + width / 2,
        DATA["3-channel"],
        width,
        label="3-channel",
        color=color_3ch,
        edgecolor=edge,
        linewidth=0.8,
        alpha=0.82,
    )

    # Highlight the selected architecture used in the main experiments.
    ltc4_idx = models.index("LTC-4")
    bars_5[ltc4_idx].set_alpha(1.0)
    bars_3[ltc4_idx].set_alpha(1.0)
    bars_5[ltc4_idx].set_linewidth(1.2)
    bars_3[ltc4_idx].set_linewidth(1.2)

    for bars in (bars_5, bars_3):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 8,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of trainable parameters", fontsize=12, fontweight="bold")
    ax.set_title(
        "LTC-RNN Parameter Count by Neuron Number",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, 455)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=11)

    fig.tight_layout()
    out_png = OUT_DIR / "ltc_parameter_sweep_bar.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(out_png)


if __name__ == "__main__":
    main()
