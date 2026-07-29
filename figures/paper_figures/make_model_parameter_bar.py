from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent

DATA = pd.DataFrame(
    {
        "Model": ["Vanilla RNN-8", "LSTM-8", "LTC-4"],
        "5-channel": [202, 538, 110],
        "3-channel": [186, 474, 86],
    }
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA.to_csv(OUT_DIR / "model_parameter_counts.csv", index=False)

    models = DATA["Model"].tolist()
    x = np.arange(len(models))
    width = 0.34

    color_5ch = "#4C78A8"
    color_3ch = "#F58518"
    edge = "#303030"

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=300)

    bars_5 = ax.bar(
        x - width / 2,
        DATA["5-channel"],
        width,
        label="5-channel",
        color=color_5ch,
        edgecolor=edge,
        linewidth=0.8,
    )
    bars_3 = ax.bar(
        x + width / 2,
        DATA["3-channel"],
        width,
        label="3-channel",
        color=color_3ch,
        edgecolor=edge,
        linewidth=0.8,
    )

    # Emphasize LTC-4 as the lightweight target model.
    ltc_idx = models.index("LTC-4")
    bars_5[ltc_idx].set_alpha(1.0)
    bars_3[ltc_idx].set_alpha(1.0)
    for i, (bar5, bar3) in enumerate(zip(bars_5, bars_3)):
        if i != ltc_idx:
            bar5.set_alpha(0.78)
            bar3.set_alpha(0.78)

    for bars in (bars_5, bars_3):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 10,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of trainable parameters", fontsize=12, fontweight="bold")
    ax.set_title("Model Parameter Count Comparison", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylim(0, 610)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=11)

    ax.annotate(
        "lightweight",
        xy=(ltc_idx, 110),
        xytext=(ltc_idx + 0.28, 210),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2),
        fontsize=11,
        fontweight="bold",
        ha="left",
    )

    fig.tight_layout()
    out_png = OUT_DIR / "model_parameter_count_bar.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(out_png)


if __name__ == "__main__":
    main()
