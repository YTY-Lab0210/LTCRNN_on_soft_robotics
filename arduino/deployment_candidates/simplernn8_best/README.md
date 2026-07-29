# Vanilla RNN-8 Arduino Candidate

此資料夾保存 Vanilla RNN-8 的 Arduino Uno deployment candidate。檔名使用 `simplernn8`，但在論文或圖表中建議稱為 `Vanilla RNN-8`。

| 檔案 | 用途 |
| --- | --- |
| `simplernn8_best.ino` | Arduino 主程式，包含 Vanilla RNN-8 權重、recurrent update、dense layer 與 softmax |
| `flash_sequence.h` | 固定 400-step 測試資料，用於 benchmark |

## 用途

此版本主要用於和 LTC-4 的 on-board inference 成本比較，例如程式大小、推論時間與記憶體使用量。實際展示若只需要 LTC-4，可不用燒錄此版本。
