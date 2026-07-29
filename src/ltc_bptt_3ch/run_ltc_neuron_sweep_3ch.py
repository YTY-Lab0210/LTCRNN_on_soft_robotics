import argparse
import gc
import glob
import os
import tempfile

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from dataset_loader_3ch import DEFAULT_DATASET_PATH, load_split_dataset


os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

TARGET_LINES = 400
CHANNELS_3CH = ["Thumb", "Middle", "Pinky"]
DEFAULT_BASE_PATH = str(DEFAULT_DATASET_PATH)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 3-channel LTC-RNN neuron sweep: LTC-1/2/4/8/16."
    )
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--units",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="LTC unit counts to evaluate.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory for CSV and figure outputs.",
    )
    return parser.parse_args()


def load_sensor_data(folder_path):
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    signal_list, label_list = [], []

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    for file_path in all_files:
        filename = os.path.basename(file_path)
        label = filename.rsplit("_", 1)[0]
        df = pd.read_csv(file_path)

        missing = [col for col in CHANNELS_3CH if col not in df.columns]
        if missing:
            raise ValueError(f"{file_path} missing columns: {missing}")

        features = df[CHANNELS_3CH].values
        if features.shape == (TARGET_LINES, len(CHANNELS_3CH)):
            signal_list.append(features)
            label_list.append(label)

    if not signal_list:
        raise ValueError(
            f"No valid samples with shape ({TARGET_LINES}, {len(CHANNELS_3CH)}) "
            f"were found in: {folder_path}"
        )

    return np.array(signal_list, dtype=np.float32), np.array(label_list)


def load_dataset(base_path):
    (x_train, y_train_raw), (x_val, y_val_raw), (x_test, y_test_raw) = load_split_dataset(base_path)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_val_encoded = encoder.transform(y_val_raw)
    y_test_encoded = encoder.transform(y_test_raw)

    num_classes = len(encoder.classes_)
    y_train = to_categorical(y_train_encoded, num_classes)
    y_val = to_categorical(y_val_encoded, num_classes)
    y_test = to_categorical(y_test_encoded, num_classes)

    print("Dataset loaded")
    print(f"  Classes: {num_classes} -> {list(encoder.classes_)}")
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    print(f"  Channels: {CHANNELS_3CH}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes


class LTCNeuron(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=keras.initializers.RandomNormal(stddev=1.0),
            trainable=True,
            name="w",
        )
        self.r = self.add_weight(
            shape=(input_dim, self.units),
            initializer="ones",
            trainable=True,
            name="r",
        )
        self.mu = self.add_weight(
            shape=(input_dim, self.units),
            initializer="zeros",
            trainable=True,
            name="mu",
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


def build_ltc_model(units, num_classes, learning_rate):
    model = keras.Sequential(
        [
            keras.Input(shape=(TARGET_LINES, len(CHANNELS_3CH))),
            keras.layers.RNN(LTCNeuron(units=units), return_sequences=False),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
    return model


def run_one_training(
    units,
    run_idx,
    data,
    num_classes,
    args,
):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data

    seed = args.seed + (units * 1000) + run_idx
    keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()

    model = build_ltc_model(units, num_classes, args.learning_rate)
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
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    best_val_acc = max(history.history.get("val_accuracy", [np.nan])) * 100.0
    epochs_trained = len(history.history.get("loss", []))

    result = {
        "Architecture": f"LTC-{units}",
        "Units": units,
        "Run": run_idx,
        "Accuracy(%)": accuracy * 100.0,
        "BestValAccuracy(%)": best_val_acc,
        "TestLoss": loss,
        "EpochsTrained": epochs_trained,
        "Channels": "3ch",
        "InputColumns": ",".join(CHANNELS_3CH),
    }

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def save_outputs(results, args):
    os.makedirs(args.out_dir, exist_ok=True)
    df_raw = pd.DataFrame(results)

    raw_path = os.path.join(args.out_dir, "LTC_Neuron_Sweep_3ch_Raw.csv")
    wide_path = os.path.join(args.out_dir, "LTC_Neuron_Sweep_3ch_Wide.csv")
    summary_path = os.path.join(args.out_dir, "LTC_Neuron_Sweep_3ch_Summary.csv")
    fig_path = os.path.join(args.out_dir, "LTC_Neuron_Sweep_3ch_Boxplot.png")

    df_raw.to_csv(raw_path, index=False)

    df_wide = df_raw.pivot(index="Run", columns="Architecture", values="Accuracy(%)")
    ordered_cols = [f"LTC-{u}" for u in args.units if f"LTC-{u}" in df_wide.columns]
    df_wide = df_wide[ordered_cols]
    df_wide.to_csv(wide_path, index=True)

    df_summary = (
        df_raw.groupby(["Architecture", "Units"], as_index=False)
        .agg(
            MeanAccuracy=("Accuracy(%)", "mean"),
            StdAccuracy=("Accuracy(%)", "std"),
            MinAccuracy=("Accuracy(%)", "min"),
            MaxAccuracy=("Accuracy(%)", "max"),
            Runs=("Accuracy(%)", "count"),
            MeanEpochs=("EpochsTrained", "mean"),
        )
        .sort_values("Units")
    )
    df_summary.to_csv(summary_path, index=False)

    plot_data = [
        df_raw.loc[df_raw["Units"] == units, "Accuracy(%)"].values
        for units in args.units
    ]
    plot_labels = [f"LTC-{units}" for units in args.units]

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=300)
    colors = ["#ccebc5", "#a8ddb5", "#7bccc4", "#4eb3d3", "#2b8cbe"]
    box = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, showmeans=True)

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#333333")

    for median in box["medians"]:
        median.set(color="#d73027", linewidth=2.0)

    for idx, values in enumerate(plot_data, start=1):
        x_jitter = np.random.normal(idx, 0.04, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            color="black",
            alpha=0.65,
            s=28,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        if len(values):
            ax.text(
                idx,
                np.min(values) - 2.0,
                f"Mean: {np.mean(values):.1f}%",
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc"),
            )

    global_min = float(np.nanmin(df_raw["Accuracy(%)"].values))
    ax.set_title("3-channel LTC-RNN Neuron Sweep (BPTT)", fontweight="bold", pad=12)
    ax.set_xlabel("Model Architecture", fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontweight="bold")
    ax.set_ylim(max(0, global_min - 10), 100)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    print("\nSaved outputs")
    print(f"  Raw:     {raw_path}")
    print(f"  Wide:    {wide_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Figure:  {fig_path}")


def main():
    args = parse_args()
    data_train, data_val, data_test, num_classes = load_dataset(args.base_path)
    data = (data_train, data_val, data_test)

    results = []
    print("\nStart 3-channel LTC neuron sweep")
    print(f"Units: {args.units}")
    print(f"Runs per model: {args.runs}")
    print(f"Epochs: {args.epochs}, patience: {args.patience}")

    for units in args.units:
        run_accuracies = []
        print(f"\n=== LTC-{units} ===")
        for run_idx in range(1, args.runs + 1):
            result = run_one_training(units, run_idx, data, num_classes, args)
            results.append(result)
            run_accuracies.append(result["Accuracy(%)"])
            print(
                f"Run {run_idx:02d}/{args.runs}: "
                f"test={result['Accuracy(%)']:.2f}% | "
                f"best_val={result['BestValAccuracy(%)']:.2f}% | "
                f"epochs={result['EpochsTrained']}"
            )
        print(f"LTC-{units} mean: {np.mean(run_accuracies):.2f}%")

    save_outputs(results, args)


if __name__ == "__main__":
    main()
