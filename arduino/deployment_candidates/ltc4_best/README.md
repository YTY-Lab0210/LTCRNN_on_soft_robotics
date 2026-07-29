# LTC-4 Arduino Candidate

此資料夾保存 LTC-4 的 Arduino Uno deployment 基本版本。

| 檔案 | 用途 |
| --- | --- |
| `ltc4_best.ino` | Arduino 主程式，包含 LTC-4 權重、z-score normalization、Euler update、dense layer、softmax、Serial Monitor 指令 |
| `flash_sequence.h` | 固定 400-step 測試資料，燒進 flash 後可用 `b`, `t`, `m`, `d` 指令測運算時間 |

## 模型設定

```text
Input channels: 5
Hidden units:   4
Classes:        10
Sequence:       400 steps
Sampling:       10 ms per step
```

## 用途

此版本主要用於檢查 LTC-4 是否能在 Arduino Uno 上完成完整 inference。若要測試繼電器 active level，建議使用 `ltc4_best_low_active/` 或 `ltc4_best_high_active/`。
