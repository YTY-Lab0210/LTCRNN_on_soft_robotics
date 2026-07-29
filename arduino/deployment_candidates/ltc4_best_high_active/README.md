# LTC-4 High-active Relay Version

此資料夾保存 LTC-4 的 high-active relay 版本。

| 檔案 | 用途 |
| --- | --- |
| `ltc4_best_high_active.ino` | Arduino 主程式，繼電器輸出邏輯為 HIGH 啟動、LOW 關閉 |
| `flash_sequence.h` | 固定 400-step 測試序列，用於 flash replay benchmark |

## Relay 邏輯

```cpp
RELAY_ON_LEVEL = HIGH
RELAY_OFF_LEVEL = LOW
```

如果你的 relay module 在 input 腳位拉高時會吸合，使用此版本。

## 實測建議

若不確定 relay 是 high-active 還是 low-active，可以分別燒錄 high-active 與 low-active 版本，先不接高壓負載，只觀察 relay 模組指示燈或吸合聲。
