# LTC-RNN Experiments: 5-Channel

This folder contains the main 5-channel training and analysis scripts.

## Main Scripts

| File | Purpose |
| --- | --- |
| `run_extended_rnn_lstm_ltc_benchmark.py` | Run the paper-style benchmark across Vanilla RNN, LSTM, and LTC models |
| `run_ltc4_few_shot_10_60_channels.py` | Run the LTC-4 few-shot experiment from 10 to 60 samples per object |
| `run_deployment_candidates.py` | Train repeated candidate models and export the best deployment weights |
| `export_ltc4_weights_for_arduino.py` | Export LTC-4 weights for Arduino sketches |
| `export_flash_sequence_for_arduino.py` | Export a fixed 400-step sequence for flash replay benchmarking |

## Notes

The default input format is the 5-channel raw ADC dataset with columns `Thumb`, `Index`, `Middle`, `Ring`, and `Pinky`. Z-score normalization should be computed from the training split only.
