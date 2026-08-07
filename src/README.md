# Source Code

This folder contains Python scripts for model training, analysis, dataset preparation, and baseline comparison. The main workflow focuses on LTC-RNN, with Vanilla RNN, LSTM, and legacy 1D-CNN baselines retained for comparison.

## Folder Overview

| Folder | Purpose |
| --- | --- |
| `ltc_bptt_5ch/` | Main 5-channel experiments, including LTC-RNN, Vanilla RNN, LSTM benchmarks, few-shot tests, deployment candidate selection, and Arduino export |
| `ltc_bptt_3ch/` | Main 3-channel experiments, using `data/dataset_new_new_new_3ch` by default |
| `baselines/` | Earlier baseline implementations for Vanilla RNN, LSTM, and 1D-CNN |
| `utils/` | Dataset cleanup, 3-channel dataset generation, z-score normalization, and train/validation/test split utilities |

## Suggested Reading Order

1. Read `ltc_bptt_5ch/README.md` for the main 5-channel experiments and deployment candidates.
2. Read `ltc_bptt_3ch/README.md` for reduced-sensor 3-channel experiments.
3. Read `baselines/README.md` for baseline model notes.
4. Read `utils/README.md` if you need to rebuild or preprocess the datasets.

## Data Format

The main raw datasets are located at:

```text
data/dataset_new_new_new/
data/dataset_new_new_new_3ch/
```

Each CSV is a 400-step time series. The 5-channel columns are `Thumb`, `Index`, `Middle`, `Ring`, and `Pinky`; the 3-channel columns are `Thumb`, `Middle`, and `Pinky`.
