from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\HAO\Desktop\YTY_from_macbook")
OUT = ROOT / "論文" / "paper_figures"

PATHS = {
    "benchmark_5ch": ROOT / "LTCRNN" / "BPTT" / "csv" / "Benchmark_Results_BPTT.csv",
    "benchmark_3ch": ROOT / "LTCRNN" / "finger_3" / "BPTT" / "csv" / "Benchmark_Results_BPTT.csv",
    "few_5ch": ROOT / "LTCRNN" / "BPTT" / "csv" / "Few_Shot_Raw_Data.csv",
    "few_3ch": ROOT / "LTCRNN" / "finger_3" / "BPTT" / "Few_Shot_Raw_Data.csv",
    "shift_5ch": ROOT / "LTCRNN" / "BPTT" / "csv" / "TimeShift_Robustness_Raw_Data.csv",
    "shift_3ch": ROOT / "LTCRNN" / "finger_3" / "BPTT" / "TimeShift_Robustness_Raw_Data.csv",
}

SERIES = {
    "5ch": "#1f77b4",
    "3ch": "#d62728",
}


def style_axes(ax, ylim=(0, 100)):
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.9)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylabel("Mean accuracy (%)", fontsize=12)


def annotate_points(ax, xs, ys, dy=1.3):
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, f"{y:.1f}", ha="center", va="bottom", fontsize=10)


def save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_benchmark():
    models = ["SimpleRNN-8", "LSTM-8", "LTC-4"]
    model_labels = ["Vanilla RNN-8", "LSTM-8", "LTC-4"]
    df5 = pd.read_csv(PATHS["benchmark_5ch"])
    df3 = pd.read_csv(PATHS["benchmark_3ch"])
    y5 = [df5[m].mean() for m in models]
    y3 = [df3[m].mean() for m in models]
    xs = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(xs, y5, marker="o", linewidth=2.7, markersize=8, label="5ch", color=SERIES["5ch"])
    ax.plot(xs, y3, marker="s", linewidth=2.7, markersize=8, label="3ch", color=SERIES["3ch"])
    annotate_points(ax, xs, y5)
    annotate_points(ax, xs, y3, dy=-4.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(model_labels)
    ax.set_title("BPTT Benchmark Mean Accuracy", fontsize=16, pad=14)
    style_axes(ax, (70, 100))
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    return save(fig, "paper_benchmark_mean_line_5ch_3ch.png")


def plot_few_shot_ltc4():
    model = "LTC-4"
    df5 = pd.read_csv(PATHS["few_5ch"])
    df3 = pd.read_csv(PATHS["few_3ch"])
    df5 = df5[df5["Model"] == model].copy()
    df3 = df3[df3["Model"] == model].copy()
    g5 = df5.groupby("Samples_Per_Class")["Accuracy"].mean().sort_index()
    g3 = df3.groupby("Samples_Per_Class")["Accuracy"].mean().sort_index()
    xs = sorted(set(g5.index).union(g3.index))
    y5 = [g5.get(x, np.nan) for x in xs]
    y3 = [g3.get(x, np.nan) for x in xs]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(xs, y5, marker="o", linewidth=2.7, markersize=8, label="5ch", color=SERIES["5ch"])
    ax.plot(xs, y3, marker="s", linewidth=2.7, markersize=8, label="3ch", color=SERIES["3ch"])
    annotate_points(ax, xs, y5)
    annotate_points(ax, xs, y3, dy=-4.2)
    ax.set_title("LTC-4 Few-shot Mean Accuracy", fontsize=16, pad=14)
    ax.set_xlabel("Training samples per class", fontsize=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    style_axes(ax, (55, 100))
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    return save(fig, "paper_ltc4_few_shot_mean_line_5ch_3ch.png")


def shift_order(values):
    def key(v):
        text = str(v)
        if text.lower() == "clean":
            return (0, 0)
        digits = "".join(ch for ch in text if ch.isdigit())
        amount = int(digits) if digits else 0
        if text.startswith("-"):
            return (-1, -amount)
        if text.startswith("+"):
            return (1, amount)
        return (2, amount)

    neg = sorted([v for v in values if str(v).startswith("-")], key=key)
    clean = [v for v in values if str(v).lower() == "clean"]
    pos = sorted([v for v in values if str(v).startswith("+")], key=key)
    other = [v for v in values if v not in neg + clean + pos]
    return neg + clean + pos + other


def plot_time_shift_ltc4():
    model = "LTC-4"
    df5 = pd.read_csv(PATHS["shift_5ch"])
    df3 = pd.read_csv(PATHS["shift_3ch"])
    df5 = df5[df5["Model"] == model].copy()
    df3 = df3[df3["Model"] == model].copy()
    col5 = "Shift Level"
    val5 = "Accuracy (%)"
    g5 = df5.groupby(col5)[val5].mean()
    g3 = df3.groupby(col5)[val5].mean()

    # The two source experiments use equivalent labels: +/-100 frames and +/-1s.
    def normalize_index(s):
        mapping = {"-100 Frames": "-1s", "+100 Frames": "+1s"}
        out = s.rename(index=mapping)
        return out

    g5 = normalize_index(g5)
    labels = shift_order(list(set(g5.index).union(g3.index)))
    y5 = [g5.get(x, np.nan) for x in labels]
    y3 = [g3.get(x, np.nan) for x in labels]
    xs = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(xs, y5, marker="o", linewidth=2.7, markersize=8, label="5ch", color=SERIES["5ch"])
    ax.plot(xs, y3, marker="s", linewidth=2.7, markersize=8, label="3ch", color=SERIES["3ch"])
    annotate_points(ax, xs, y5)
    annotate_points(ax, xs, y3, dy=-4.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_title("LTC-4 Time-shift Mean Accuracy", fontsize=16, pad=14)
    ax.set_xlabel("Temporal shift condition", fontsize=12)
    style_axes(ax, (55, 100))
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    return save(fig, "paper_ltc4_time_shift_mean_line_5ch_3ch.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_benchmark(),
        plot_few_shot_ltc4(),
        plot_time_shift_ltc4(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
