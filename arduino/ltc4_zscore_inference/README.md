# Early LTC-4 Z-score Inference Sketch

此資料夾保存早期 LTC-4 z-score inference Arduino sketch。

| 檔案 | 用途 |
| --- | --- |
| `ltc4_zscore_inference.ino` | 早期 Arduino LTC-4 推論測試程式 |

## 與 deployment_candidates 的差異

`deployment_candidates/` 內的版本較完整，包含 flash replay benchmark、relay control variants、timing breakdown 與候選模型權重。此資料夾主要保留作為早期測試紀錄。

## 建議

若要實際去手臂平台測試，優先使用：

```text
arduino/deployment_candidates/ltc4_best_low_active/
arduino/deployment_candidates/ltc4_best_high_active/
```
