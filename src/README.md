# Source Code 說明

本資料夾保存 Python 訓練、分析、資料整理與 baseline 比較程式。主要研究流程以 LTC-RNN 為核心，並保留 Vanilla RNN、LSTM、1D-CNN 等比較模型。

## 資料夾總覽

| 資料夾 | 用途 |
| --- | --- |
| `ltc_bptt_5ch/` | 5-channel 主實驗，包含 LTC-RNN、Vanilla RNN、LSTM benchmark、few-shot、部署候選模型與 Arduino 匯出 |
| `ltc_bptt_3ch/` | 3-channel 主實驗，預設讀取 `data/dataset_new_new_new_3ch` |
| `baselines/` | 早期 baseline 模型程式，包含 Vanilla RNN、LSTM、1D-CNN |
| `utils/` | 資料集整理、3-channel dataset 產生、z-score、min-max 與資料切分工具 |

## 建議閱讀順序

1. 先看 `ltc_bptt_5ch/README.md`：理解主要 5-channel 實驗與 deployment candidate 來源。
2. 再看 `ltc_bptt_3ch/README.md`：理解 reduced-sensor 3-channel 實驗。
3. 若要看比較模型，讀 `baselines/README.md`。
4. 若要重新整理 dataset，讀 `utils/README.md`。

## 資料格式提醒

目前 repo 內主要 raw dataset 位於：

```text
data/dataset_new_new_new/
data/dataset_new_new_new_3ch/
```

每筆 CSV 為 400-step time-series。5-channel 欄位為 `Thumb`, `Index`, `Middle`, `Ring`, `Pinky`；3-channel 欄位為 `Thumb`, `Middle`, `Pinky`。
