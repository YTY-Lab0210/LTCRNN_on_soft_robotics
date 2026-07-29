# 3-channel LTC-RNN 程式說明

此資料夾保存 3-channel 實驗程式。輸入通道為：

```text
Thumb, Middle, Pinky
```

目前預設讀取 repo 內的新資料集：

```text
data/dataset_new_new_new_3ch/
```

`dataset_loader_3ch.py` 會自動將 1000 筆平面 CSV 依類別切成 train / validation / test，預設為每類 60 / 20 / 20，也會用 training set 計算 z-score mean/std，再套用到 validation/test。

## 主要程式

| 檔案 | 用途 |
| --- | --- |
| `dataset_loader_3ch.py` | 共用 3-channel dataset loader，自動切分資料並執行 z-score normalization |
| `run_ltc_neuron_sweep_3ch.py` | 主要 3-channel LTC 神經元數 sweep，測試 LTC-1/2/4/8/16 |
| `run_bench_mark_bptt.py` | 3-channel benchmark，早期比較 1D-CNN、Vanilla RNN-8、LSTM-8、LTC-4 |
| `run_few_shot_efficiency.py` | 3-channel few-shot 測試，觀察少量 training samples 對辨識率的影響 |
| `run_time_shift.py` | 3-channel time-shift robustness 測試 |
| `LTC_4neuron.py` | 單獨訓練 3-channel LTC-4，並輸出訓練曲線與模型參數 |

## 繪圖程式

| 檔案 | 用途 |
| --- | --- |
| `draw_benchmark.py` | 由 `Benchmark_Results_BPTT.csv` 重畫 benchmark 圖 |
| `draw_time_shift.py` | 由 `TimeShift_Robustness_Raw_Data.csv` 重畫 time-shift 圖 |

## 輸出檔案

| 檔案 / 資料夾 | 內容 |
| --- | --- |
| `Benchmark_Results_BPTT.csv` | benchmark raw result |
| `Few_Shot_Raw_Data.csv` | few-shot raw result |
| `LTC_Neuron_Sweep_3ch_*.csv` | LTC neuron sweep raw / summary / wide format |
| `csv/` | 備份或整理後的 CSV 結果 |
| `picture/` | 部分圖表輸出 |

## 建議重跑順序

1. 先執行 `dataset_loader_3ch.py` 的簡單匯入測試，確認 dataset 可讀。
2. 跑 `run_ltc_neuron_sweep_3ch.py` 取得 LTC-1/2/4/8/16 結果。
3. 需要比較模型時，再跑 `run_bench_mark_bptt.py`。
4. 需要少樣本或時間平移分析時，再跑 few-shot / time-shift 程式。

## 注意

`run_bench_mark_bptt.py`、`run_few_shot_efficiency.py` 與 `run_time_shift.py` 是較早期的研究腳本，但已改成吃新的 `dataset_new_new_new_3ch`。其中 `1D-CNN` 屬於早期 baseline，若目前簡報或 paper 不需要，可以只看 LTC、Vanilla RNN、LSTM 結果。
