from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\HAO\Desktop\YTY_from_macbook")
FIVE_CH_CSV = ROOT / "LTCRNN" / "BPTT" / "csv" / "Few_Shot_Raw_Data.csv"
THREE_CH_CSV = ROOT / "LTCRNN" / "finger_3" / "BPTT" / "Few_Shot_Raw_Data.csv"
OUT_DIR = Path(__file__).resolve().parent

MODELS = ["LTC-4", "SimpleRNN-8", "LSTM-8"]
MODEL_LABELS = {
    "LTC-4": "LTC-RNN",
    "SimpleRNN-8": "Vanilla RNN",
    "LSTM-8": "LSTM",
}
COLORS = {
    "LTC-4": "#1f77b4",
    "SimpleRNN-8": "#6f6f6f",
    "LSTM-8": "#d62728",
}
MARKERS = {
    "LTC-4": "o",
    "SimpleRNN-8": "s",
    "LSTM-8": "^",
}
END_LABEL_OFFSETS = {
    "LTC-4": 0.45,
    "SimpleRNN-8": -0.45,
    "LSTM-8": 0.05,
}


def load_summary(csv_path: Path, channel_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["Model"].isin(MODELS)].copy()
    summary = (
        df.groupby(["Model", "Samples_Per_Class"], as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            StdAccuracy=("Accuracy", "std"),
            Runs=("Accuracy", "count"),
        )
        .sort_values(["Model", "Samples_Per_Class"])
    )
    summary["Channel"] = channel_label
    return summary


def plot_panel(ax, summary: pd.DataFrame, title: str) -> None:
    for model in MODELS:
        sub = summary[summary["Model"] == model].sort_values("Samples_Per_Class")
        x = sub["Samples_Per_Class"].to_numpy()
        y = sub["MeanAccuracy"].to_numpy()

        ax.plot(
            x,
            y,
            marker=MARKERS[model],
            markersize=6,
            linewidth=2.4,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )

        ax.text(
            x[-1] + 0.8,
            y[-1] + END_LABEL_OFFSETS[model],
            f"{y[-1]:.1f}",
            va="center",
            fontsize=9,
            color=COLORS[model],
            fontweight="bold",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Training samples per class", fontsize=11, fontweight="bold")
    ax.set_xticks([3, 6, 15, 30, 60])
    ax.set_xlim(1, 66)
    ax.set_ylim(58, 96)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    five = load_summary(FIVE_CH_CSV, "5-channel")
    three = load_summary(THREE_CH_CSV, "3-channel")
    combined = pd.concat([five, three], ignore_index=True)

    summary_csv = OUT_DIR / "paper_few_shot_models_5ch_3ch_summary.csv"
    export_combined = combined.copy()
    export_combined["Model"] = export_combined["Model"].replace(
        {"SimpleRNN-8": "Vanilla RNN-8"}
    )
    export_combined.to_csv(summary_csv, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.35), dpi=300, sharey=True)
    plot_panel(axes[0], five, "5-channel input")
    plot_panel(axes[1], three, "3-channel input")
    axes[0].set_ylabel("Test accuracy (%)", fontsize=11, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1.0])

    out_png = OUT_DIR / "paper_few_shot_models_5ch_3ch_line.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(out_png)
    print(summary_csv)


if __name__ == "__main__":
    main()
