from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(r"C:\Users\HAO\Desktop\YTY_from_macbook")
FIG_DIR = ROOT / "\u8ad6\u6587" / "paper_figures"
EXT_DIR = FIG_DIR / "extended_benchmark"

EXT_SUMMARY = EXT_DIR / "extended_benchmark_summary.csv"
LTC_SWEEP_SUMMARY = FIG_DIR / "ltc_neuron_sweep_5ch_3ch_summary.csv"
LTC_PARAM_COUNTS = FIG_DIR / "ltc_parameter_sweep_counts.csv"

OUT_CSV = EXT_DIR / "benchmark_accuracy_by_family_with_ltc8_ltc16_summary.csv"
OUT_5CH = EXT_DIR / "benchmark_accuracy_by_family_with_ltc8_ltc16_5ch.png"
OUT_3CH = EXT_DIR / "benchmark_accuracy_by_family_with_ltc8_ltc16_3ch.png"
OUT_BOTH = EXT_DIR / "benchmark_accuracy_by_family_with_ltc8_ltc16_both.png"


FAMILY_ORDER = {
    "Vanilla RNN": ["SimpleRNN-4", "SimpleRNN-8", "SimpleRNN-16"],
    "LSTM": ["LSTM-4", "LSTM-8", "LSTM-16"],
    "LTC": ["LTC-4", "LTC-8", "LTC-16"],
}

FAMILY_COLORS = {
    "Vanilla RNN": "#8A8F98",
    "LSTM": "#C35668",
    "LTC": "#2F7D8C",
}


def model_family(model: str) -> str:
    if model.startswith("SimpleRNN"):
        return "Vanilla RNN"
    if model.startswith("LSTM"):
        return "LSTM"
    if model.startswith("LTC"):
        return "LTC"
    raise ValueError(f"Unknown model family: {model}")


def model_units(model: str) -> int:
    return int(model.split("-")[-1])


def load_combined_summary() -> pd.DataFrame:
    extended = pd.read_csv(EXT_SUMMARY)
    extended = extended[extended["Model"].isin(FAMILY_ORDER["Vanilla RNN"] + FAMILY_ORDER["LSTM"])].copy()
    extended["Family"] = extended["Model"].map(model_family)
    extended["Units"] = extended["Model"].map(model_units)
    extended["Params"] = extended["TrainableParams"].astype(int)
    extended["Source"] = "extended benchmark"
    extended = extended[
        [
            "Channel",
            "Family",
            "Model",
            "Units",
            "Params",
            "MeanAccuracy",
            "StdAccuracy",
            "Runs",
            "Source",
        ]
    ]

    ltc_sweep = pd.read_csv(LTC_SWEEP_SUMMARY)
    ltc_params = pd.read_csv(LTC_PARAM_COUNTS).set_index("Model")
    ltc_rows = []
    for _, row in ltc_sweep[ltc_sweep["Architecture"].isin(FAMILY_ORDER["LTC"])].iterrows():
        model = row["Architecture"]
        for channel in ["5ch", "3ch"]:
            suffix = "5ch" if channel == "5ch" else "3ch"
            param_col = "5-channel" if channel == "5ch" else "3-channel"
            ltc_rows.append(
                {
                    "Channel": channel,
                    "Family": "LTC",
                    "Model": model,
                    "Units": int(row["Units"]),
                    "Params": int(ltc_params.loc[model, param_col]),
                    "MeanAccuracy": float(row[f"MeanAccuracy_{suffix}"]),
                    "StdAccuracy": float(row[f"StdAccuracy_{suffix}"]),
                    "Runs": int(row[f"Runs_{suffix}"]),
                    "Source": "LTC neuron sweep",
                }
            )

    combined = pd.concat([extended, pd.DataFrame(ltc_rows)], ignore_index=True)
    combined["FamilyOrder"] = combined["Family"].map({name: i for i, name in enumerate(FAMILY_ORDER)})
    combined["ModelOrder"] = combined.apply(
        lambda row: FAMILY_ORDER[row["Family"]].index(row["Model"]),
        axis=1,
    )
    combined = combined.sort_values(["Channel", "FamilyOrder", "ModelOrder"]).drop(
        columns=["FamilyOrder", "ModelOrder"]
    )
    return combined


def positions_for_models():
    positions = {}
    family_centers = {}
    x = 0.0
    for family, models in FAMILY_ORDER.items():
        start = x
        for model in models:
            positions[model] = x
            x += 0.72
        end = x - 0.72
        family_centers[family] = (start + end) / 2
        x += 0.72
    return positions, family_centers


def apply_axes_style(ax, channel: str):
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 100)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_title(f"Model Benchmark ({channel})", pad=12, fontsize=18, fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=1.2, length=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def draw_channel(data: pd.DataFrame, channel: str, out_path: Path, *, figsize=(10.4, 5.2)):
    channel_data = data[data["Channel"] == channel].copy()
    positions, family_centers = positions_for_models()
    channel_data["x"] = channel_data["Model"].map(positions)
    channel_data["Color"] = channel_data["Family"].map(FAMILY_COLORS)

    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    ax.bar(
        channel_data["x"],
        channel_data["MeanAccuracy"],
        width=0.46,
        color=channel_data["Color"],
        edgecolor="#222222",
        linewidth=1.0,
    )

    ax.set_xticks(
        [positions[model] for family in FAMILY_ORDER.values() for model in family],
        [str(model_units(model)) for family in FAMILY_ORDER.values() for model in family],
        rotation=0,
    )
    apply_axes_style(ax, channel)

    for family, center in family_centers.items():
        ax.text(
            center,
            -0.15,
            family,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            color=FAMILY_COLORS[family],
            fontweight="bold",
            clip_on=False,
        )

    ax.set_xlim(-0.55, max(positions.values()) + 0.55)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.26, top=0.86)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_both(data: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=240, sharey=True)
    for ax, channel in zip(axes, ["5ch", "3ch"]):
        channel_data = data[data["Channel"] == channel].copy()
        positions, family_centers = positions_for_models()
        channel_data["x"] = channel_data["Model"].map(positions)
        channel_data["Color"] = channel_data["Family"].map(FAMILY_COLORS)
        ax.bar(
            channel_data["x"],
            channel_data["MeanAccuracy"],
            width=0.46,
            color=channel_data["Color"],
            edgecolor="#222222",
            linewidth=1.0,
        )
        ax.set_xticks(
            [positions[model] for family in FAMILY_ORDER.values() for model in family],
            [str(model_units(model)) for family in FAMILY_ORDER.values() for model in family],
            rotation=0,
        )
        apply_axes_style(ax, channel)
        for family, center in family_centers.items():
            ax.text(
                center,
                -0.15,
                family,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                color=FAMILY_COLORS[family],
                fontweight="bold",
                clip_on=False,
            )
        ax.set_xlim(-0.55, max(positions.values()) + 0.55)

    axes[1].set_ylabel("")
    legend = [
        Patch(facecolor=FAMILY_COLORS[family], edgecolor="#222222", label=family)
        for family in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.26, top=0.80, wspace=0.16)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    combined = load_combined_summary()
    export_combined = combined.copy()
    export_combined["Model"] = export_combined["Model"].str.replace(
        "SimpleRNN", "Vanilla RNN", regex=False
    )
    export_combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    draw_channel(combined, "5ch", OUT_5CH)
    draw_channel(combined, "3ch", OUT_3CH)
    draw_both(combined, OUT_BOTH)

    print(OUT_CSV)
    print(OUT_5CH)
    print(OUT_3CH)
    print(OUT_BOTH)


if __name__ == "__main__":
    main()
