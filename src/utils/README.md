# Utility Scripts 說明

此資料夾保存資料整理與前處理工具。

| 檔案 | 用途 |
| --- | --- |
| `create_3ch_dataset.py` | 從 5-channel raw CSV 產生 3-channel dataset，只保留 `Time_ms`, `Thumb`, `Middle`, `Pinky`，並在每類內隨機重新編號 |
| `split_data.py` | 將資料切成 training / validation / test 的輔助工具 |
| `z_score.py` | 對資料做 z-score normalization 的工具 |
| `minmax.py` | 對資料做 min-max normalization 的工具 |
| `test_slow.py` | 早期測試或除錯用的小工具 |

## 目前最常用的工具

若要重新產生 3-channel dataset：

```bash
python src/utils/create_3ch_dataset.py
```

預設會讀取：

```text
data/dataset_new_new_new/
```

並輸出：

```text
data/dataset_new_new_new_3ch/
data/dataset_new_new_new_3ch_manifest.csv
```

## Z-score 原則

正式訓練時應只用 training set 計算 mean/std，再套用到 validation 與 test。這個原則也已用在 `src/ltc_bptt_3ch/dataset_loader_3ch.py`。
