# Lightweight Tactile Object Recognition for a Soft Robotic Hand

This repository contains the code, dataset, result-plot scripts, and Arduino deployment sketches for a lightweight tactile object recognition system based on Liquid Time-Constant RNNs (LTC-RNNs).

The project focuses on flex-sensor time-series signals collected from a five-finger soft robotic hand. The main model is an LTC-RNN trained for tactile object classification, with Vanilla RNN and LSTM baselines for comparison.

## Repository Structure

```text
data/
  dataset_new_new_new/            Main renamed 10-class raw dataset
  dataset_new_new_new_3ch/        Randomly renumbered 3-channel raw dataset

src/
  ltc_bptt_5ch/                   LTC-RNN training, benchmark, few-shot, and deployment export scripts
  ltc_bptt_3ch/                   3-channel LTC-RNN experiments
  ga_5ch/                         Genetic algorithm and hybrid optimization scripts for 5-channel experiments
  ga_3ch/                         Genetic algorithm and hybrid optimization scripts for 3-channel experiments
  baselines/
    vanillarnn/                   Vanilla RNN baseline scripts
    lstm/                         LSTM baseline scripts
    cnn1d/                        1D-CNN legacy baseline scripts
  utils/                          Data split, z-score, min-max, and visualization utilities

arduino/
  deployment_candidates/          Arduino Uno deployment sketches for LTC-4, Vanilla RNN-8, and LSTM-8
  ltc4_zscore_inference/          Earlier LTC-4 z-score inference sketch

figures/
  paper_figures/                  Scripts and generated figures used for paper-style visualization

docs/
  EXCLUDED_FILES.md               Notes about intentionally excluded private or temporary files
```

## Dataset

The main dataset is under:

```text
data/dataset_new_new_new/
```

It contains the renamed raw ADC sequences using the same display labels as the confusion-matrix figures, such as `Baseball`, `Bottle`, `Sponge Dice`, and `3D-Printed Part`.

The current dataset has 999 CSV files: 99 `Baseball` samples and 100 samples for each of the other nine classes. See `data/README.md` for the full class list and counts.

A 3-channel version is also provided:

```text
data/dataset_new_new_new_3ch/
```

This version keeps `Time_ms`, `Thumb`, `Middle`, and `Pinky`, then randomly renumbers files within each class. The source-to-output mapping is stored in:

```text
data/dataset_new_new_new_3ch_manifest.csv
```

## Python Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, seaborn, SciPy, and Numba.

## Main Experiments

Important entry points:

```text
src/ltc_bptt_5ch/run_extended_rnn_lstm_ltc_benchmark.py
src/ltc_bptt_5ch/run_ltc4_few_shot_10_60_channels.py
src/ltc_bptt_5ch/run_deployment_candidates.py
src/ltc_bptt_3ch/run_ltc_neuron_sweep_3ch.py
```

Paper-style figure generation scripts are in:

```text
figures/paper_figures/
```

## Arduino Deployment

The Arduino sketches are prepared for Arduino Uno. Flex sensors are read from `A0` to `A4`, and relay control signals use digital pins `2` to `6`.

Two LTC-4 relay variants are included:

```text
arduino/deployment_candidates/ltc4_best_low_active/
arduino/deployment_candidates/ltc4_best_high_active/
```

Use the low-active version when the relay turns on with `LOW`. Use the high-active version when the relay turns on with `HIGH`. See `arduino/README.md` for the serial commands and test procedure.

## Notes

This repository intentionally excludes old dataset backups, temporary model caches, private thesis forms, signed documents, and teacher draft files. See `docs/EXCLUDED_FILES.md`.
