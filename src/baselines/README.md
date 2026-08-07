# Baseline Models

This folder stores earlier baseline implementations used to compare against LTC-RNN:

```text
Vanilla RNN
LSTM
1D-CNN
```

## Folders

| Folder | Purpose |
| --- | --- |
| `vanillarnn/` | Vanilla RNN / SimpleRNN baseline scripts |
| `lstm/` | LSTM baseline scripts |
| `cnn1d/` | Legacy 1D-CNN baseline scripts |

## Usage Note

Most scripts in this folder come from earlier experiments, and some may still contain legacy local paths or older dataset assumptions. For the current paper-style benchmark, prefer:

```text
src/ltc_bptt_5ch/run_extended_rnn_lstm_ltc_benchmark.py
```

That script integrates Vanilla RNN, LSTM, and LTC models and is the recommended entry point for reproducing the current benchmark results.
