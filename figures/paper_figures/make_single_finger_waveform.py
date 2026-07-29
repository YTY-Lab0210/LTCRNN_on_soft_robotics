from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
DATA_FILE = (
    Path(r"C:\Users\HAO\Desktop\YTY_from_macbook")
    / "dataset_xx2020_new_new_new"
    / "dataset_602020"
    / "training"
    / "cylinder_037.csv"
)
CHANNEL = "Index"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE).iloc[:400].copy()
    time_s = df["Time_ms"] / 1000.0
    signal = df[CHANNEL]
    smooth = signal.rolling(window=7, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=300)

    ax.plot(time_s, signal, color="#9aa7b2", linewidth=1.0, alpha=0.42, label="Raw signal")
    ax.plot(time_s, smooth, color="#1f77b4", linewidth=2.8, label="Smoothed signal")

    ax.set_xlim(0, 4)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1 s", "2 s", "3 s", "4 s"], fontsize=11, fontweight="bold")

    y_margin = (signal.max() - signal.min()) * 0.14
    ax.set_ylim(signal.min() - y_margin, signal.max() + y_margin)
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Flex sensor output (ADC value)", fontsize=12, fontweight="bold")
    ax.set_title("Single-Finger Grasping Signal", fontsize=15, fontweight="bold", pad=12)

    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)

    ax.annotate(
        "grasp-induced\nbending response",
        xy=(3.0, smooth.iloc[int(3.0 / 0.01)]),
        xytext=(2.05, signal.max() - y_margin * 0.45),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.1),
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
    )

    fig.tight_layout()
    out_png = OUT_DIR / "single_finger_grasp_waveform_adc.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    out_csv = OUT_DIR / "single_finger_grasp_waveform_adc.csv"
    pd.DataFrame(
        {
            "Time_s": time_s,
            f"{CHANNEL}_ADC": signal,
            f"{CHANNEL}_ADC_smoothed": smooth,
        }
    ).to_csv(out_csv, index=False)

    print(out_png)


if __name__ == "__main__":
    main()
