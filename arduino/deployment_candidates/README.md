# Arduino Deployment Candidates

此資料夾保存可燒錄到 Arduino Uno 的候選模型程式。每個子資料夾通常包含：

```text
*.ino              Arduino 主程式
flash_sequence.h   固定 400-step 測試序列，用於 flash replay benchmark
```

## 子資料夾

| 資料夾 | 用途 |
| --- | --- |
| `ltc4_best/` | LTC-4 on-board inference 基本版本 |
| `ltc4_best_low_active/` | LTC-4，繼電器為 LOW active 時使用 |
| `ltc4_best_high_active/` | LTC-4，繼電器為 HIGH active 時使用 |
| `simplernn8_best/` | Vanilla RNN-8 Arduino deployment candidate |
| `lstm8_best/` | LSTM-8 Arduino deployment candidate |

## Serial Monitor 指令

常用 baud rate：

```text
115200
```

常用指令：

| 指令 | 用途 |
| --- | --- |
| `g` | 實際讀取 flex sensor，執行一次 400-sample grasp window |
| `r` | 釋放所有繼電器 |
| `b` | 使用 `flash_sequence.h` 裡的固定資料做一次 benchmark |
| `t` | 印出一次 task-level timing breakdown |
| `m` | 重複 flash benchmark 100 次 |
| `d` | 重複 timing breakdown 100 次 |

## 實際抓握流程

```text
輸入 g
  -> 每 10 ms 讀取一次 A0-A4
  -> 前 1.5 s 為 pre-grasp baseline
  -> 1.5 s 後啟動 D2-D6 繼電器
  -> 收滿 400 點
  -> z-score normalization
  -> model inference
  -> 印出 predicted class / confidence / probabilities
  -> 關閉繼電器
```

## 接線提醒

Arduino 只接低壓控制側：

```text
A0-A4  -> flex sensor 分壓輸出
D2-D6  -> relay module control input
GND    -> relay control-side GND 共地
```

泵浦、電磁閥與 110 V 負載側不可接到 Arduino 邏輯腳位。
