from pathlib import Path

import numpy as np
import pandas as pd


TARGET_LINES = 400
CHANNELS_3CH = ["Thumb", "Middle", "Pinky"]
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "dataset_new_new_new_3ch"
)


def load_sensor_data(folder_path, target_lines=TARGET_LINES, channels=CHANNELS_3CH):
    folder = Path(folder_path)
    all_files = sorted(folder.glob("*.csv"))
    signal_list, label_list = [], []

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    for file_path in all_files:
        label = file_path.stem.rsplit("_", 1)[0]
        df = pd.read_csv(file_path)

        missing = [col for col in channels if col not in df.columns]
        if missing:
            raise ValueError(f"{file_path} missing columns: {missing}")

        features = df[channels].to_numpy(dtype=np.float32)
        if features.shape == (target_lines, len(channels)):
            signal_list.append(features)
            label_list.append(label)

    if not signal_list:
        raise ValueError(
            f"No valid samples with shape ({target_lines}, {len(channels)}) "
            f"were found in: {folder}"
        )

    return np.asarray(signal_list, dtype=np.float32), np.asarray(label_list)


def split_by_class(x, y, train_count=60, val_count=20, test_count=20, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in sorted(np.unique(y)):
        cls_idx = np.where(y == cls)[0]
        cls_idx = np.array(cls_idx, copy=True)
        rng.shuffle(cls_idx)

        required = train_count + val_count + test_count
        if len(cls_idx) < required:
            raise ValueError(
                f"Class {cls!r} has {len(cls_idx)} samples; "
                f"{required} are required for {train_count}/{val_count}/{test_count} split."
            )

        train_idx.extend(cls_idx[:train_count])
        val_idx.extend(cls_idx[train_count : train_count + val_count])
        test_idx.extend(cls_idx[train_count + val_count : required])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])


def zscore_from_train(x_train, x_val, x_test):
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.where(std == 0, 1.0, std)

    def normalize(x):
        return ((x - mean) / std).astype(np.float32)

    return normalize(x_train), normalize(x_val), normalize(x_test), mean.astype(np.float32), std.astype(np.float32)


def load_split_dataset(
    base_path=DEFAULT_DATASET_PATH,
    train_count=60,
    val_count=20,
    test_count=20,
    seed=42,
    normalize=True,
):
    base = Path(base_path)

    if all((base / name).is_dir() for name in ["training", "validation", "test"]):
        x_train, y_train = load_sensor_data(base / "training")
        x_val, y_val = load_sensor_data(base / "validation")
        x_test, y_test = load_sensor_data(base / "test")
    else:
        x_all, y_all = load_sensor_data(base)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_by_class(
            x_all,
            y_all,
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            seed=seed,
        )

    if normalize:
        x_train, x_val, x_test, mean, std = zscore_from_train(x_train, x_val, x_test)
        print(f"Z-score mean: {mean.tolist()}")
        print(f"Z-score std:  {std.tolist()}")

    print("3-channel dataset loaded")
    print(f"  Dataset path: {base}")
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    print(f"  Classes: {sorted(np.unique(y_train).tolist())}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
