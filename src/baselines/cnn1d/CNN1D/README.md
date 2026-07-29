# 1D-CNN Scripts

此資料夾保存早期 1D-CNN baseline 與相關測試。

| 檔案 | 用途 |
| --- | --- |
| `cnn_1d.py` | 訓練基本 1D-CNN sequence classifier |
| `cnn_1d_add_noise.py` | 測試加入雜訊後 1D-CNN 的分類表現 |
| `cnn_1d_earlystopping_comparison.py` | 比較 early stopping 對 1D-CNN 訓練的影響 |
| `cnn_1d_reduce_trainingset.py` | 測試減少 training samples 時 1D-CNN 的表現 |
| `loss.py` | 繪製或分析 loss curve |
| `structure.py` | 輸出或檢視 1D-CNN 模型結構 |

## 注意

這些程式是早期 baseline，部分路徑與資料夾名稱可能需要手動更新才能重跑目前 dataset。
