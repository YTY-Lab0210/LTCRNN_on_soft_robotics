import argparse
import glob
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


TARGET_LINES = 400

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset_xx2020_new_new_new" / "dataset_602020_zscore"
DEFAULT_OUT_DIR = ROOT / "\u8ad6\u6587" / "paper_figures" / "extended_benchmark"

CHANNELS = {
    "5ch": ["Thumb", "Index", "Middle", "Ring", "Pinky"],
    "3ch": ["Thumb", "Middle", "Pinky"],
}

MODEL_ORDER = [
    "SimpleRNN-4",
    "SimpleRNN-8",
    "SimpleRNN-16",
    "LSTM-4",
    "LSTM-8",
    "LSTM-16",
    "LTC-4",
]


class LTCNeuron(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        init_w = keras.initializers.RandomNormal(stddev=1.0)

        self.w = self.add_weight(
            name="w", shape=(input_dim, self.units), initializer=init_w, trainable=True
        )
        self.r = self.add_weight(
            name="r", shape=(input_dim, self.units), initializer="ones", trainable=True
        )
        self.mu = self.add_weight(
            name="mu", shape=(input_dim, self.units), initializer="zeros", trainable=True
        )

    def call(self, inputs, states):
        x = states[0]
        delta_t = 0.01

        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        sigma = tf.math.sigmoid(inputs_expanded * self.r + self.mu)
        damping = 1.0 + tf.reduce_sum(tf.abs(self.w) * sigma, axis=1)
        driving = tf.reduce_sum(self.w * sigma, axis=1)

        dx = -damping * x + driving
        x_new = x + delta_t * dx

        return x_new, [x_new]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def parse_model_name(model_name):
    family, units_text = model_name.split("-")
    return family, int(units_text)


def load_sensor_data(folder_path, columns):
    files = sorted(glob.glob(str(Path(folder_path) / "*.csv")))
    signals = []
    labels = []

    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    for file_path in files:
        label = Path(file_path).stem.rsplit("_", 1)[0]
        df = pd.read_csv(file_path)
        features = df[columns].to_numpy(dtype=np.float32)

        if features.shape == (TARGET_LINES, len(columns)):
            signals.append(features)
            labels.append(label)

    if not signals:
        raise RuntimeError(
            f"No valid samples with shape ({TARGET_LINES}, {len(columns)}) in {folder_path}"
        )

    return np.asarray(signals, dtype=np.float32), np.asarray(labels)


def load_channel_dataset(base_path, channel):
    columns = CHANNELS[channel]
    x_train, y_train_raw = load_sensor_data(Path(base_path) / "training", columns)
    x_val, y_val_raw = load_sensor_data(Path(base_path) / "validation", columns)
    x_test, y_test_raw = load_sensor_data(Path(base_path) / "test", columns)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_val_encoded = encoder.transform(y_val_raw)
    y_test_encoded = encoder.transform(y_test_raw)
    num_classes = len(encoder.classes_)

    y_train = keras.utils.to_categorical(y_train_encoded, num_classes)
    y_val = keras.utils.to_categorical(y_val_encoded, num_classes)
    y_test = keras.utils.to_categorical(y_test_encoded, num_classes)

    return {
        "columns": columns,
        "num_classes": num_classes,
        "class_names": encoder.classes_.tolist(),
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_encoded": y_test_encoded,
    }


def build_model(model_name, input_shape, num_classes):
    family, units = parse_model_name(model_name)

    if family == "SimpleRNN":
        recurrent_layer = keras.layers.SimpleRNN(units, return_sequences=False)
    elif family == "LSTM":
        recurrent_layer = keras.layers.LSTM(units, return_sequences=False)
    elif family == "LTC":
        recurrent_layer = keras.layers.RNN(LTCNeuron(units=units), return_sequences=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            recurrent_layer,
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name=model_name.replace("-", "_"),
    )


def append_result(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    pd.DataFrame([row]).to_csv(
        csv_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def existing_keys(raw_path):
    if not raw_path.exists():
        return set()

    df = pd.read_csv(raw_path)
    if df.empty:
        return set()

    return set(zip(df["Channel"], df["Model"], df["Run"]))


def run_one_model(dataset, channel, model_name, run_index, args):
    keras.backend.clear_session()
    keras.utils.set_random_seed(args.seed + run_index)

    input_shape = (TARGET_LINES, len(dataset["columns"]))
    model = build_model(model_name, input_shape, dataset["num_classes"])
    trainable_params = int(model.count_params())

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm)
    model.compile(
        optimizer=optimizer,
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
        dataset["x_train"],
        dataset["y_train"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(dataset["x_val"], dataset["y_val"]),
        callbacks=[early_stopping],
        verbose=args.verbose,
    )

    test_loss, test_accuracy = model.evaluate(dataset["x_test"], dataset["y_test"], verbose=0)
    probabilities = model.predict(dataset["x_test"], verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    macro_f1 = f1_score(dataset["y_test_encoded"], y_pred, average="macro", zero_division=0)

    history_df = pd.DataFrame(history.history)
    best_idx = int(history_df["val_loss"].idxmin())

    return {
        "Channel": channel,
        "Model": model_name,
        "Run": run_index,
        "TrainableParams": trainable_params,
        "Accuracy(%)": test_accuracy * 100.0,
        "Macro-F1(%)": macro_f1 * 100.0,
        "TestLoss": float(test_loss),
        "BestValAccuracy(%)": float(history_df.loc[best_idx, "val_accuracy"] * 100.0),
        "BestValLoss": float(history_df.loc[best_idx, "val_loss"]),
        "EpochsTrained": int(len(history_df)),
        "InputColumns": ",".join(dataset["columns"]),
    }


def summarize(raw_path, summary_path, models, channels):
    raw = pd.read_csv(raw_path)
    raw["Model"] = pd.Categorical(raw["Model"], categories=models, ordered=True)
    raw["Channel"] = pd.Categorical(raw["Channel"], categories=channels, ordered=True)

    summary = (
        raw.groupby(["Channel", "Model"], observed=True)
        .agg(
            Runs=("Run", "count"),
            TrainableParams=("TrainableParams", "first"),
            MeanAccuracy=("Accuracy(%)", "mean"),
            StdAccuracy=("Accuracy(%)", "std"),
            MeanMacroF1=("Macro-F1(%)", "mean"),
            StdMacroF1=("Macro-F1(%)", "std"),
            MeanTestLoss=("TestLoss", "mean"),
        )
        .reset_index()
        .sort_values(["Channel", "Model"])
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def style_axes(ax):
    ax.set_ylim(60, 100)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.grid(False)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)


def plot_channel(summary, channel, out_dir):
    data = summary[summary["Channel"].astype(str) == channel].copy()
    if data.empty:
        return None

    data["Model"] = pd.Categorical(data["Model"], categories=MODEL_ORDER, ordered=True)
    data = data.sort_values("Model")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "hatch.linewidth": 1.5,
        }
    )

    fig, ax = plt.subplots(figsize=(11.6, 5.5), dpi=220)
    x = np.arange(len(data))
    bar_w = 0.22
    gap = 0.14

    accuracy_color = "#c85a6a"
    f1_color = "#efb2bd"
    edge_color = "#9b2f47"

    ax.bar(
        x - (bar_w + gap) / 2,
        data["MeanAccuracy"],
        width=bar_w,
        color=accuracy_color,
        edgecolor=edge_color,
        linewidth=1.0,
        label="Accuracy",
    )
    ax.bar(
        x + (bar_w + gap) / 2,
        data["MeanMacroF1"],
        width=bar_w,
        color=f1_color,
        edgecolor=edge_color,
        linewidth=1.0,
        hatch="----",
        label="F1-score",
    )

    ax.set_title(f"Extended Benchmark ({channel})", pad=12)
    ax.set_ylabel("Accuracy and F1-score (%)")
    ax.set_xticks(x, data["Model"].astype(str), rotation=24, ha="right")
    ax.set_xlim(-0.55, len(data) - 0.45)
    style_axes(ax)

    legend_handles = [
        Patch(facecolor=accuracy_color, edgecolor=edge_color, label="Accuracy"),
        Patch(facecolor=f1_color, edgecolor=edge_color, hatch="----", label="F1-score"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    out_path = out_dir / f"extended_benchmark_accuracy_f1_{channel}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SimpleRNN/LSTM/LTC benchmark and draw accuracy/F1 bar charts."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--channels", choices=["5ch", "3ch", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--jit-compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    selected_channels = ["5ch", "3ch"] if args.channels == "both" else [args.channels]
    selected_models = args.models

    raw_path = args.out_dir / "extended_benchmark_raw.csv"
    summary_path = args.out_dir / "extended_benchmark_summary.csv"

    if not args.plot_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        done = existing_keys(raw_path) if args.resume else set()

        for channel in selected_channels:
            dataset = load_channel_dataset(args.dataset, channel)
            print(
                f"\n[{channel}] samples: train={len(dataset['x_train'])}, "
                f"val={len(dataset['x_val'])}, test={len(dataset['x_test'])}, "
                f"classes={dataset['num_classes']}"
            )

            for model_name in selected_models:
                for run_index in range(1, args.runs + 1):
                    key = (channel, model_name, run_index)
                    if key in done:
                        print(f"Skip {channel} {model_name} run {run_index:02d} (already exists)")
                        continue

                    print(f"Run {channel} {model_name} {run_index:02d}/{args.runs} ...", end=" ")
                    row = run_one_model(dataset, channel, model_name, run_index, args)
                    append_result(raw_path, row)
                    print(
                        f"Acc={row['Accuracy(%)']:.2f}%, "
                        f"F1={row['Macro-F1(%)']:.2f}%, "
                        f"epochs={row['EpochsTrained']}"
                    )

    if not raw_path.exists():
        raise FileNotFoundError(f"No raw results found: {raw_path}")

    summary = summarize(raw_path, summary_path, selected_models, selected_channels)
    print(f"\nSaved summary: {summary_path}")

    for channel in selected_channels:
        png_path = plot_channel(summary, channel, args.out_dir)
        if png_path is not None:
            print(f"Saved chart: {png_path}")


if __name__ == "__main__":
    main()
