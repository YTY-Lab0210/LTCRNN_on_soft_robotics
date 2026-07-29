# Vanilla RNN Baseline

此資料夾保存 Vanilla RNN baseline 的訓練與繪圖程式。程式內部使用 Keras `SimpleRNN` layer；論文或圖表中建議標示為 `Vanilla RNN`。

## 檔案

| 檔案 / 資料夾 | 用途 |
| --- | --- |
| `SimpleRNN/` | 早期 Vanilla RNN 訓練腳本 |
| `draw_simplernn.py` | 將 Vanilla RNN 訓練或測試結果畫成圖 |

## 注意

`SimpleRNN/` 內的程式是早期版本，可能需要手動更新 `BASE_PATH` 才能吃目前 repo 內的 dataset。若只是要取得目前 benchmark 結果，建議使用 `src/ltc_bptt_5ch/run_extended_rnn_lstm_ltc_benchmark.py`。
