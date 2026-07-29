from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\HAO\Desktop\YTY_from_macbook")
OUT_DIR = Path(__file__).resolve().parent
SOURCE_TABLE_DIR = OUT_DIR / "source_tables"

CSV_5CH = (
    SOURCE_TABLE_DIR
    / "ltc4_bptt_5ch_confusion_matrices.csv"
)
CSV_3CH = (
    SOURCE_TABLE_DIR
    / "ltc4_bptt_3ch_confusion_matrices.csv"
)

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
X_LABELS = [
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


def mean_matrix(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df[df["method"] == METHOD].copy()
    grouped = (
        df.groupby(["true_label", "predicted_label"], as_index=False)["row_percent"]
        .mean()
    )
    matrix = (
        grouped.pivot(
            index="true_label",
            columns="predicted_label",
            values="row_percent",
        )
        .reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return matrix


def format_cell(value: float) -> str:
    if value < 1.0:
        return ""
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f}"
    return f"{value:.1f}"


def draw(matrix: np.ndarray, channel_label: str, out_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.7, 7.2), dpi=300)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="equal")

    ax.set_title(
        f"BPTT checkpoint 50-run mean ({channel_label})\nrow-normalized (%)",
        fontsize=20,
        pad=16,
    )
    ax.set_xlabel("Predicted label", fontsize=16, labelpad=14)
    ax.set_ylabel("True label", fontsize=16, labelpad=14)

    ticks = np.arange(len(CLASS_ORDER))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(X_LABELS, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(CLASS_ORDER, fontsize=11)

    ax.set_xticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = format_cell(value)
            if not label:
                continue
            color = "white" if value >= 55 else "#1f2933"
            ax.text(j, i, label, ha="center", va="center", fontsize=10, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)

    out_path = OUT_DIR / out_name
    fig.subplots_adjust(left=0.18, bottom=0.24, right=0.9, top=0.86)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out5 = draw(mean_matrix(CSV_5CH), "5ch", "bptt_mean_confusion_5ch.png")
    out3 = draw(mean_matrix(CSV_3CH), "3ch", "bptt_mean_confusion_3ch.png")
    print(out5)
    print(out3)


if __name__ == "__main__":
    main()
