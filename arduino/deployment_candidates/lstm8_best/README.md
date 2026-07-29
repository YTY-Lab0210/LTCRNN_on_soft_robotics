# LSTM-8 Arduino Candidate

此資料夾保存 LSTM-8 的 Arduino Uno deployment candidate。

| 檔案 | 用途 |
| --- | --- |
| `lstm8_best.ino` | Arduino 主程式，包含 LSTM-8 gate weights、state update、dense layer 與 softmax |
| `flash_sequence.h` | 固定 400-step 測試資料，用於 benchmark |

## 用途

此版本主要用於和 LTC-4、Vanilla RNN-8 比較 on-board inference 成本。LSTM 權重與 gate 計算較多，通常比 LTC-4 更吃記憶體與運算時間。

## 注意

Arduino Uno SRAM 很小，實測時若遇到不穩定、重開機或輸出亂碼，可能和記憶體壓力有關。
