# Baseline Models 說明

此資料夾保存早期 baseline 模型程式，用於和 LTC-RNN 比較。包含：

```text
Vanilla RNN
LSTM
1D-CNN
```

## 資料夾

| 資料夾 | 用途 |
| --- | --- |
| `vanillarnn/` | Vanilla RNN / SimpleRNN baseline |
| `lstm/` | LSTM baseline |
| `cnn1d/` | 1D-CNN legacy baseline |

## 使用提醒

這些 baseline 程式多數是早期實驗版本，部分檔案內仍保留舊電腦或舊資料集路徑。若要正式重跑目前 repo 的資料，建議優先使用：

```text
src/ltc_bptt_5ch/run_extended_rnn_lstm_ltc_benchmark.py
```

該檔案已整合 Vanilla RNN、LSTM 與 LTC 的 benchmark，比較適合作為目前 paper / GitHub 的主線結果來源。
