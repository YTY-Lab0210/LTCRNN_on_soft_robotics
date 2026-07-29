# Arduino Deployment 說明

本資料夾保存 Arduino Uno 端的 on-board inference 測試程式。主要目標是讓 Arduino 直接讀取 flex sensor、控制五個繼電器啟動抓握，並在板子上完成 LTC-4 推論。

## 腳位設定

Flex sensor 類比輸入：

```text
A0, A1, A2, A3, A4
```

Relay control 輸出腳：

```text
D2, D3, D4, D5, D6
```

Serial Monitor baud rate：

```text
115200
```

## LTC-4 繼電器版本

提供兩種版本，用來測試不同 relay module 的觸發邏輯：

```text
deployment_candidates/ltc4_best_low_active/ltc4_best_low_active.ino
deployment_candidates/ltc4_best_high_active/ltc4_best_high_active.ino
```

Low-active：

```cpp
RELAY_ON_LEVEL = LOW
RELAY_OFF_LEVEL = HIGH
```

High-active：

```cpp
RELAY_ON_LEVEL = HIGH
RELAY_OFF_LEVEL = LOW
```

這兩份程式沒有自動判斷 relay 型態；實測時直接燒錄對應版本即可。

## Serial Monitor 指令

```text
g  開始一次 400-sample grasp window
r  釋放所有繼電器
b  使用 flash-stored 400-sample window 做 inference benchmark
t  顯示單次 task-level timing breakdown
m  重複 flash benchmark 100 次
d  重複 timing breakdown 100 次
```

## `g` 指令流程

```text
輸入 g
  -> 每 10 ms 取樣一次 flex sensor
  -> 前 1.5 s 為抓握前狀態
  -> 1.5 s 後五個繼電器啟動，開始抓握
  -> 收滿 400 點
  -> 執行 z-score normalization
  -> 執行 LTC-4 Euler update
  -> 執行 dense layer 與 softmax
  -> 印出分類結果
  -> 關閉繼電器並釋放
```

## Flash replay benchmark

`b`, `t`, `m`, `d` 指令使用燒在程式中的 `flash_sequence.h` 測試固定 400-step 序列。這個模式不等待實際 10 ms 取樣，因此可用來估計純 on-board inference 的運算時間。

目前 timing breakdown 的階段包含：

```text
State reset
Flash read
Z-score normalize
LTC Euler update
Dense + softmax
Argmax
Total
```

## 接線提醒

Arduino 只處理低壓訊號：

```text
Arduino analog pins -> flex sensor 分壓輸出
Arduino digital pins -> relay module input pins
Arduino GND -> relay control-side GND
```

Relay module 需要自己的合適 DC 供電。泵浦、電磁閥與 110 V 電源側不可直接接到 Arduino 邏輯腳位；高壓與負載端應維持隔離，只透過 relay contact side 控制。
