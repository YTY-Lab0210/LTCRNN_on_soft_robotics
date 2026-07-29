from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(r"C:\Users\HAO\Desktop\YTY_from_macbook")
OUT_DIR = ROOT / "\u8ad6\u6587" / "paper_figures"

CONFUSION_5CH = (
    ROOT
    / "LTCRNN"
    / "GA"
    / "outputs"
    / "LTC4_BPTT_ES200_MaxEpoch2000_GA_TrainFit_ConstrainedValSelect_eps0p005_Restart3_Run50"
    / "LTC4_BPTT_ES200_MaxEpoch2000_GA_TrainFit_ConstrainedValSelect_eps0p005_Restart3_Run50_ConfusionMatrices.csv"
)
CONFUSION_3CH = (
    ROOT
    / "LTCRNN"
    / "finger_3"
    / "GA"
    / "outputs"
    / "LTC4_3CH_BPTT_ES200_MaxEpoch2000_GA_TrainFit_ConstrainedValSelect_eps0p005_Restart3_Run50"
    / "LTC4_3CH_BPTT_ES200_MaxEpoch2000_GA_TrainFit_ConstrainedValSelect_eps0p005_Restart3_Run50_ConfusionMatrices.csv"
)
BENCHMARK_SUMMARY = OUT_DIR / "extended_benchmark" / "extended_benchmark_summary.csv"

OUT_CONFUSION = OUT_DIR / "paper_confusion_no_numbers_5ch_3ch.png"
OUT_CONFUSION_IDS = OUT_DIR / "paper_confusion_no_numbers_5ch_3ch_ids.png"
OUT_BENCHMARK = OUT_DIR / "paper_benchmark_accuracy_f1_compact.png"

METHOD = "BPTT checkpoint"
CLASS_ORDER = [
    "Baseball",
    "Bottle",
    "Sponge Dice",
    "Tape",
    "Plush Toy",
    "Optical Mouse",
    "Smartphone",
    "Rubik's Cube",
    "Stuffed Ball",
    "3D-Printed Part",
]
CLASS_TICKS = [
    "Baseball",
    "Bottle",
    "Sponge\nDice",
    "Tape",
    "Plush\nToy",
    "Optical\nMouse",
    "Smartphone",
    "Rubik's\nCube",
    "Stuffed\nBall",
    "3D-Printed\nPart",
]

MODEL_GROUPS = {
    "Vanilla RNN": ["SimpleRNN-4", "SimpleRNN-8", "SimpleRNN-16"],
    "LSTM": ["LSTM-4", "LSTM-8", "LSTM-16"],
    "LTC": ["LTC-4"],
}
MODELS = [model for group in MODEL_GROUPS.values() for model in group]


def mean_confusion_matrix(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df[df["method"] == METHOD].copy()
    grouped = (
        df.groupby(["true_label", "predicted_label"], as_index=False)["row_percent"]
        .mean()
    )
    return (
        grouped.pivot(index="true_label", columns="predicted_label", values="row_percent")
        .reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def draw_confusion_no_numbers():
    matrices = [
        ("5-channel", mean_confusion_matrix(CONFUSION_5CH)),
        ("3-channel", mean_confusion_matrix(CONFUSION_3CH)),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), dpi=450)
    images = []

    for ax, (title, matrix) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        images.append(im)
        ax.set_title(title, pad=5)
        ax.set_xlabel("Predicted label", labelpad=4)
        ax.set_ylabel("True label", labelpad=4)

        ticks = np.arange(len(CLASS_ORDER))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(CLASS_TICKS, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(CLASS_TICKS)

        ax.set_xticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=2.2, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    cbar = fig.colorbar(images[0], ax=axes, fraction=0.032, pad=0.035)
    cbar.set_label("Row-normalized accuracy (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2.2, width=0.7)

    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.27, top=0.88, wspace=0.28)
    fig.savefig(OUT_CONFUSION, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_confusion_no_numbers_ids():
    matrices = [
        ("(a) 5-channel", mean_confusion_matrix(CONFUSION_5CH)),
        ("(b) 3-channel", mean_confusion_matrix(CONFUSION_3CH)),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.85), dpi=500)
    images = []
    ticks = np.arange(len(CLASS_ORDER))
    tick_labels = [str(i) for i in range(1, len(CLASS_ORDER) + 1)]

    for idx, (ax, (title, matrix)) in enumerate(zip(axes, matrices)):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        images.append(im)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel("Predicted class", labelpad=2)
        if idx == 0:
            ax.set_ylabel("True class", labelpad=2)
        else:
            ax.set_yticklabels([])

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        ax.set_xticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=2.0, width=0.65)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    cbar_ax = fig.add_axes([0.91, 0.21, 0.018, 0.66])
    cbar = fig.colorbar(images[0], cax=cbar_ax)
    cbar.set_label("Row-normalized accuracy (%)", fontsize=7.3)
    cbar.ax.tick_params(labelsize=6.8, length=2.0, width=0.65)

    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.18, top=0.89, wspace=0.16)
    fig.savefig(OUT_CONFUSION_IDS, facecolor="white")
    plt.close(fig)


def draw_compact_benchmark():
    summary = pd.read_csv(BENCHMARK_SUMMARY)
    rows = []
    for channel in ["3ch", "5ch"]:
        for model in MODELS:
            row = summary[(summary["Channel"] == channel) & (summary["Model"] == model)].iloc[0]
            rows.append(
                {
                    "Channel": channel,
                    "Model": model,
                    "Accuracy": float(row["MeanAccuracy"]),
                    "Macro-F1": float(row["MeanMacroF1"]),
                }
            )
    data = pd.DataFrame(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "hatch.linewidth": 1.0,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.85), dpi=450, sharey=True)

    colors = {"3ch": "#efb2bd", "5ch": "#c85a6a"}
    hatches = {"3ch": "////", "5ch": None}
    edge = "#6f1d32"

    x_positions = {}
    group_centers = {}
    x_cursor = 0.0
    for group_name, models in MODEL_GROUPS.items():
        start = x_cursor
        for model in models:
            x_positions[model] = x_cursor
            x_cursor += 0.72
        end = x_cursor - 0.72
        group_centers[group_name] = (start + end) / 2.0
        x_cursor += 0.58

    x = np.array([x_positions[model] for model in MODELS], dtype=float)
    width = 0.24

    for ax, metric, panel in zip(axes, ["Accuracy", "Macro-F1"], ["(a)", "(b)"]):
        for offset, channel in [(-width / 2, "3ch"), (width / 2, "5ch")]:
            subset = data[data["Channel"] == channel].set_index("Model").loc[MODELS]
            ax.bar(
                x + offset,
                subset[metric].to_numpy(dtype=float),
                width=width,
                color=colors[channel],
                edgecolor=edge,
                linewidth=0.7,
                hatch=hatches[channel],
                label=channel.replace("ch", "-channel"),
            )

        ax.set_title(f"{panel} {metric}", loc="left", pad=4)
        ax.set_xticks(x, [model.split("-")[-1] for model in MODELS])
        ax.set_ylim(60, 100)
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.tick_params(axis="both", length=2.5, width=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
        for group_name, center in group_centers.items():
            ax.text(
                center,
                -0.16,
                group_name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.4,
                color="#4b4b4b",
                fontweight="bold" if group_name == "LTC" else "normal",
            )
        ax.set_xlim(min(x) - 0.50, max(x) + 0.50)

    axes[0].set_ylabel("Score (%)")

    legend_handles = [
        Patch(facecolor=colors["3ch"], edgecolor=edge, hatch=hatches["3ch"], label="3-channel"),
        Patch(facecolor=colors["5ch"], edgecolor=edge, label="5-channel"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.03),
        ncol=2,
    )

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.28, top=0.78, wspace=0.14)
    fig.savefig(OUT_BENCHMARK, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_confusion_no_numbers()
    draw_confusion_no_numbers_ids()
    draw_compact_benchmark()
    print(OUT_CONFUSION)
    print(OUT_CONFUSION_IDS)
    print(OUT_BENCHMARK)


if __name__ == "__main__":
    main()
