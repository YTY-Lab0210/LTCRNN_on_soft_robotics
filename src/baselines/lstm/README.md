# LSTM Baseline

此資料夾保存 LSTM baseline 的訓練與繪圖程式，用於與 LTC-RNN 和 Vanilla RNN 比較。

## 檔案

| 檔案 / 資料夾 | 用途 |
| --- | --- |
| `LSTM/` | 早期 LSTM 訓練腳本 |
| `draw_lstm.py` | 將 LSTM 訓練或測試結果畫成圖 |

## 注意

`LSTM/` 內的程式是早期版本，可能需要手動更新 `BASE_PATH`。目前正式 benchmark 建議使用 `src/ltc_bptt_5ch/run_extended_rnn_lstm_ltc_benchmark.py`。
