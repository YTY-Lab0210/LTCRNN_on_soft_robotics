# LTC-4 Low-active Relay Version

此資料夾保存 LTC-4 的 low-active relay 版本。

| 檔案 | 用途 |
| --- | --- |
| `ltc4_best_low_active.ino` | Arduino 主程式，繼電器輸出邏輯為 LOW 啟動、HIGH 關閉 |
| `flash_sequence.h` | 固定 400-step 測試序列，用於 flash replay benchmark |

## Relay 邏輯

```cpp
RELAY_ON_LEVEL = LOW
RELAY_OFF_LEVEL = HIGH
```

如果你的 relay module 在 input 腳位拉低時會吸合，使用此版本。

## 實測建議

先不要接負載，只用 Serial Monitor 送 `r` 與 `g`，確認 D2-D6 輸出邏輯與 relay 模組反應正確，再接泵浦與電磁閥。
