import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.patches import Patch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

import run_extended_rnn_lstm_ltc_benchmark as bench


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset_xx2020_new_new_new" / "dataset_602020_zscore"
DEFAULT_OUT_DIR = ROOT / "\u8ad6\u6587" / "paper_figures" / "few_shot_10_60"
MODEL_NAME = "LTC-4"
SAMPLE_LEVELS = [10, 20, 30, 40, 50, 60]
CHANNEL_ORDER = ["5ch", "3ch"]
CHANNEL_LABELS = {"5ch": "5 channel", "3ch": "3 channel"}
CHANNEL_COLORS = {"5ch": "#aa3048", "3ch": "#df9aa8"}


def load_channel_arrays(base_path, channel):
    columns = bench.CHANNELS[channel]
    x_train, y_train_raw = bench.load_sensor_data(Path(base_path) / "training", columns)
    x_val, y_val_raw = bench.load_sensor_data(Path(base_path) / "validation", columns)
    x_test, y_test_raw = bench.load_sensor_data(Path(base_path) / "test", columns)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_val_encoded = encoder.transform(y_val_raw)
    y_test_encoded = encoder.transform(y_test_raw)
    num_classes = len(encoder.classes_)

    return {
        "columns": columns,
        "num_classes": num_classes,
        "class_names": encoder.classes_.tolist(),
        "x_train": x_train,
        "y_train_raw": y_train_raw,
        "y_train_encoded": y_train_encoded,
        "x_val": x_val,
        "y_val": keras.utils.to_categorical(y_val_encoded, num_classes),
        "x_test": x_test,
        "y_test": keras.utils.to_categorical(y_test_encoded, num_classes),
        "y_test_encoded": y_test_encoded,
    }


def few_shot_indices(y_raw, samples_per_class, rng):
    selected = []
    selected_counts = {}
    for label in np.unique(y_raw):
        idx = np.where(y_raw == label)[0]
        take_count = min(samples_per_class, len(idx))
        selected_counts[str(label)] = int(take_count)
        if len(idx) <= samples_per_class:
            selected.extend(idx.tolist())
        else:
            selected.extend(rng.choice(idx, samples_per_class, replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64), selected_counts


def existing_keys(raw_path):
    if not raw_path.exists():
        return set()
    df = pd.read_csv(raw_path)
    if df.empty:
        return set()
    return set(zip(df["Channel"], df["Samples_Per_Class"], df["Run"]))


def append_result(raw_path, row):
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        raw_path,
        mode="a",
        header=not raw_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def run_one(dataset, channel, samples_per_class, run_index, args):
    keras.backend.clear_session()
    seed = args.seed + (100000 if channel == "3ch" else 0) + samples_per_class * 100 + run_index
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    idx, selected_counts = few_shot_indices(dataset["y_train_raw"], samples_per_class, rng)
    x_train = dataset["x_train"][idx]
    y_train_encoded = dataset["y_train_encoded"][idx]
    y_train = keras.utils.to_categorical(y_train_encoded, dataset["num_classes"])

    model = bench.build_model(
        MODEL_NAME,
        (bench.TARGET_LINES, len(dataset["columns"])),
        dataset["num_classes"],
    )
    trainable_params = int(model.count_params())
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            clipnorm=args.clipnorm,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=args.jit_compile,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=min(args.batch_size, len(x_train)),
        validation_data=(dataset["x_val"], dataset["y_val"]),
        callbacks=[early_stopping],
        verbose=args.verbose,
    )

    test_loss, test_accuracy = model.evaluate(dataset["x_test"], dataset["y_test"], verbose=0)
    probs = model.predict(dataset["x_test"], verbose=0)
    y_pred = np.argmax(probs, axis=1)
    macro_f1 = f1_score(dataset["y_test_encoded"], y_pred, average="macro", zero_division=0)

    history_df = pd.DataFrame(history.history)
    best_idx = int(history_df["val_loss"].idxmin())

    return {
        "Channel": channel,
        "ChannelLabel": CHANNEL_LABELS[channel],
        "Model": MODEL_NAME,
        "Samples_Per_Class": samples_per_class,
        "Run": run_index,
        "MinSelectedSamplesPerClass": min(selected_counts.values()),
        "TotalSelectedTrainingSamples": len(idx),
        "TrainableParams": trainable_params,
        "Accuracy": float(test_accuracy * 100.0),
        "MacroF1": float(macro_f1 * 100.0),
        "TestLoss": float(test_loss),
        "BestValAccuracy": float(history_df.loc[best_idx, "val_accuracy"] * 100.0),
        "BestValLoss": float(history_df.loc[best_idx, "val_loss"]),
        "EpochsTrained": int(len(history_df)),
        "InputColumns": ",".join(dataset["columns"]),
    }


def summarize(raw_path, summary_path):
    raw = pd.read_csv(raw_path)
    summary = (
        raw.groupby(["Channel", "ChannelLabel", "Model", "Samples_Per_Class"], observed=True)
        .agg(
            Runs=("Run", "count"),
            MeanAccuracy=("Accuracy", "mean"),
            StdAccuracy=("Accuracy", "std"),
            MeanMacroF1=("MacroF1", "mean"),
            StdMacroF1=("MacroF1", "std"),
            MeanEpochsTrained=("EpochsTrained", "mean"),
            TrainableParams=("TrainableParams", "first"),
        )
        .reset_index()
    )
    channel_order = {name: i for i, name in enumerate(CHANNEL_ORDER)}
    summary["_channel_order"] = summary["Channel"].map(channel_order)
    summary = summary.sort_values(["_channel_order", "Samples_Per_Class"]).drop(columns="_channel_order")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def style_axes(ax, y_min):
    ax.set_xlim(0, 70)
    ax.set_ylim(75, 100)
    ax.set_xticks(SAMPLE_LEVELS)
    ax.set_yticks([75, 80, 85, 90, 95, 100])
    ax.grid(True, color="#d2d2d2", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.6)
        spine.set_color("#111111")
    ax.tick_params(axis="both", width=1.4, length=6, color="#111111", labelsize=14)
    ax.set_xlabel("training samples per object", fontsize=17, fontweight="bold")
    ax.set_ylabel("accuracy [%]", fontsize=17, fontweight="bold")


def plot_paper_style(summary, out_dir, y_min=None):
    data_min = float(summary["MeanAccuracy"].min())
    if y_min is None:
        y_min = 80 if data_min >= 79.5 else 70

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
    )

    fig, ax = plt.subplots(figsize=(4.25, 4.45), dpi=300)
    for channel in CHANNEL_ORDER:
        sub = summary[summary["Channel"] == channel].sort_values("Samples_Per_Class")
        ax.plot(
            sub["Samples_Per_Class"],
            sub["MeanAccuracy"],
            color=CHANNEL_COLORS[channel],
            marker="o",
            markersize=6.4,
            markeredgewidth=0,
            linewidth=2.2,
            label=CHANNEL_LABELS[channel],
        )

    style_axes(ax, y_min)
    legend_handles = [
        Patch(facecolor=CHANNEL_COLORS["5ch"], edgecolor=CHANNEL_COLORS["5ch"], label=CHANNEL_LABELS["5ch"]),
        Patch(facecolor=CHANNEL_COLORS["3ch"], edgecolor=CHANNEL_COLORS["3ch"], label=CHANNEL_LABELS["3ch"]),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=True,
        fancybox=False,
        edgecolor="#111111",
        framealpha=1.0,
        facecolor="white",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.42),
        borderaxespad=0.0,
        fontsize=13,
    )

    fig.tight_layout(pad=0.5)
    out_png = out_dir / "paper_ltc4_few_shot_10_60_accuracy.png"
    out_svg = out_dir / "paper_ltc4_few_shot_10_60_accuracy.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png, out_svg


def parse_args():
    parser = argparse.ArgumentParser(description="Run and plot LTC-4 few-shot 5ch/3ch sample-efficiency results.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--channels", choices=["5ch", "3ch", "both"], default="both")
    parser.add_argument("--samples", nargs="+", type=int, default=SAMPLE_LEVELS)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--jit-compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--ylim-min", type=float, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "paper_ltc4_few_shot_10_60_raw.csv"
    summary_path = args.out_dir / "paper_ltc4_few_shot_10_60_summary.csv"

    selected_channels = CHANNEL_ORDER if args.channels == "both" else [args.channels]
    selected_samples = list(args.samples)

    if not args.plot_only:
        done = existing_keys(raw_path) if args.resume else set()
        for channel in selected_channels:
            dataset = load_channel_arrays(args.dataset, channel)
            print(
                f"\n[{channel}] train={len(dataset['x_train'])}, "
                f"val={len(dataset['x_val'])}, test={len(dataset['x_test'])}, "
                f"classes={dataset['num_classes']}"
            )
            for samples_per_class in selected_samples:
                for run_index in range(1, args.runs + 1):
                    key = (channel, samples_per_class, run_index)
                    if key in done:
                        print(f"Skip {channel} samples={samples_per_class} run {run_index:02d}")
                        continue
                    print(
                        f"Run {channel} {MODEL_NAME} samples={samples_per_class} "
                        f"{run_index:02d}/{args.runs} ...",
                        end=" ",
                        flush=True,
                    )
                    row = run_one(dataset, channel, samples_per_class, run_index, args)
                    append_result(raw_path, row)
                    print(
                        f"Acc={row['Accuracy']:.2f}%, "
                        f"F1={row['MacroF1']:.2f}%, "
                        f"epochs={row['EpochsTrained']}"
                    )

    if not raw_path.exists():
        raise FileNotFoundError(f"No raw results found: {raw_path}")

    summary = summarize(raw_path, summary_path)
    out_png, out_svg = plot_paper_style(summary, args.out_dir, args.ylim_min)
    print(f"Saved raw: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved PNG: {out_png}")
    print(f"Saved SVG: {out_svg}")


if __name__ == "__main__":
    main()
