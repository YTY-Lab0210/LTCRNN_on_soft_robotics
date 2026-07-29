# 5-channel LTC-RNN / Benchmark 程式說明

此資料夾保存 5-channel 主實驗程式。輸入通道為五根手指：

```text
Thumb, Index, Middle, Ring, Pinky
```

主要用途是訓練 LTC-RNN，並與 Vanilla RNN、LSTM 等模型比較，再挑選可部署到 Arduino Uno 的候選權重。

## 主要程式

| 檔案 | 用途 |
| --- | --- |
| `run_extended_rnn_lstm_ltc_benchmark.py` | 主要 benchmark 程式，比較 Vanilla RNN-4/8/16、LSTM-4/8/16、LTC-4，輸出 accuracy、Macro-F1 與參數量摘要 |
| `run_ltc4_few_shot_10_60_channels.py` | 少樣本實驗，測試每類訓練樣本數 10 到 60 時，5-channel 與 3-channel 的 LTC-4 表現 |
| `run_deployment_candidates.py` | 從多次訓練中挑出 LTC-4、LSTM-8、Vanilla RNN-8 的較佳 run，並輸出 Arduino deployment 需要的權重 |
| `export_ltc4_weights_for_arduino.py` | 將 LTC-4 權重轉成 Arduino C/C++ 陣列格式 |
| `export_flash_sequence_for_arduino.py` | 將一筆 400-step CSV 序列轉成 `flash_sequence.h`，用於 Arduino flash replay benchmark |

## LTC-RNN 訓練與分析

| 檔案 | 用途 |
| --- | --- |
| `LTC_4neuron.py` | 單獨訓練 LTC-4，並印出模型參數與訓練曲線 |
| `LTC_8neuron.py` | 單獨訓練 LTC-8 |
| `LTC_everyneuron_boxplot.py` | 早期 LTC 不同神經元數比較與 boxplot |
| `run_training_curve.py` | 產生模型訓練過程曲線 |

## 繪圖程式

| 檔案 | 用途 |
| --- | --- |
| `draw_from_csv.py` | 從 CSV 結果重畫圖 |
| `draw_benchmark_constraint.py` | 繪製 benchmark / constraint 結果 |
| `draw_few_data_boxplot.py` | 繪製 few-shot boxplot |
| `draw_training_curve.py` | 繪製訓練曲線 |
| `repair_few_shot_10_60_csv.py` | 修補 few-shot CSV 格式，供後續 paper figure 腳本使用 |

## 輸出資料夾

| 資料夾 | 內容 |
| --- | --- |
| `csv/` | benchmark、few-shot 與主要模型比較的 CSV |
| `diagram/` | 早期與主要模型比較相關的圖表輸出 |

## 注意

部分舊程式仍保留早期實驗用的檔名或輸出位置；若要重跑最新 GitHub dataset，建議先確認 `DEFAULT_DATASET` 或 `BASE_PATH` 是否指向 repo 內的 `data/dataset_new_new_new/`。
