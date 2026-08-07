# LTC-RNN Experiments: 3-Channel

This folder contains reduced-sensor experiments using the 3-channel dataset.

## Main Scripts

| File | Purpose |
| --- | --- |
| `dataset_loader_3ch.py` | Load the 3-channel dataset |
| `run_ltc_neuron_sweep_3ch.py` | Run LTC-1/2/4/8/16 neuron-count experiments |
| `run_bench_mark_bptt.py` | Run baseline model comparisons for the 3-channel setting |
| `run_few_shot_efficiency.py` | Run reduced-data experiments for the 3-channel setting |
| `LTC_4neuron.py` | Standalone LTC-4 training script |

## Dataset

The expected dataset path is:

```text
data/dataset_new_new_new_3ch/
```

Columns are `Time_ms`, `Thumb`, `Middle`, and `Pinky`.
