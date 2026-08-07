# Paper Figure Scripts

This folder contains paper and presentation figures, redraw scripts, compact result tables, and generated images.

## Main Redraw Scripts

| File | Purpose |
| --- | --- |
| `make_paper_compact_figures.py` | Generate compact benchmark figures and confusion matrices without cell values |
| `redraw_mean_confusion_full_labels.py` | Generate confusion matrices with full class labels |
| `make_accuracy_f1_bar_bptt_channels.py` | Generate 3-channel / 5-channel Accuracy and Macro-F1 bar charts |
| `make_extended_benchmark_accuracy_by_family.py` | Group Vanilla RNN, LSTM, and LTC models by family in the benchmark chart |
| `make_ltc_neuron_sweep_5ch_3ch_bar.py` | Generate LTC-1/2/4/8/16 bar charts for 5-channel and 3-channel inputs |
| `make_ltc_neuron_sweep_5ch_3ch_line.py` | Generate the LTC neuron-sweep line chart |
| `make_few_shot_models_5ch_3ch.py` | Generate few-shot training-sample curves |
| `make_model_parameter_bar.py` | Generate model parameter-count comparison chart |
| `make_ltc_parameter_sweep_bar.py` | Generate LTC parameter-count comparison across neuron counts |

## Image and Schematic Scripts

| File | Purpose |
| --- | --- |
| `make_single_finger_waveform.py` | Generate a single-finger grasp ADC waveform |
| `remove_flex_sensor_background.py` | Remove the background from the flex-sensor image |
| `remove_soft_finger_background.py` | Remove the background from the soft-finger image |

## Folders

| Folder | Contents |
| --- | --- |
| `source_tables/` | Compact BPTT result tables used by figure scripts |
| `extended_benchmark/` | Extended benchmark figures and summary CSV files |
| `few_shot_10_60/` | Figures and CSV files for 10 to 60 training samples per object |
| `extended_benchmark_smoke/` | Smoke-test or shortened benchmark outputs |

## Common Output Figures

| Figure | Purpose |
| --- | --- |
| `paper_benchmark_accuracy_f1_compact.png` | Paper-style model benchmark |
| `paper_confusion_no_numbers_5ch_3ch_ids.png` | Compact confusion matrix using class IDs |
| `ltc_neuron_sweep_5ch_3ch_bar.png` | LTC neuron-number sweep |
| `model_parameter_count_bar.png` | Model parameter-count comparison |
