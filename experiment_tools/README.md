# Experiment tools

本資料夾整理軟式機器人實驗使用的資料收集、感測器測試、relay 測試與相機角度同步程式。
所有 Python 程式都移除了本機絕對路徑與固定 USB port；未指定 `--port` 時會嘗試自動尋找 Arduino。

## 內容

| 功能 | Arduino firmware | Python 程式 |
|---|---|---|
| 五指 flex sensor，100 Hz、400 點 | `arduino/flex_sensor_5ch/flex_sensor_5ch.ino` | `python/collect_flex_five_channel.py` |
| 單一 flex ADC stream | `arduino/flex_adc/flex_adc.ino` | `python/read_flex_adc.py` |
| 單一 flex 電阻值 | `arduino/flex_resistance/flex_resistance.ino` | `python/collect_flex_resistance.py` |
| 五路 relay 循環測試 | `arduino/relay_test/relay_test.ino` | 不需要 |
| 三路 FSR | `arduino/fsr_3ch/fsr_3ch.ino` | `python/collect_fsr.py` |
| Drop/catch 文字紀錄解析 | 使用會輸出對應文字的實驗 firmware | `python/collect_drop_timing.py` |
| 相機測試 | 不需要 | `python/camera_test.py` |
| 相機黑膠帶角度＋單一 flex ADC 同步 | `arduino/flex_adc/flex_adc.ino` | `python/record_flex_angle.py` |

## 安裝

在 repository 根目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r experiment_tools/requirements.txt
```

Windows 啟用虛擬環境：

```powershell
.venv\Scripts\activate
```

列出可用 serial port：

```bash
python -m serial.tools.list_ports
```

## 常用指令

五指 flex sensor，每次收 400 點：

```bash
python experiment_tools/python/collect_flex_five_channel.py \
  --port /dev/cu.usbmodem101 \
  --label bottle \
  --samples 400
```

如果只接一個 Arduino，通常可省略 `--port`：

```bash
python experiment_tools/python/collect_flex_five_channel.py --label bottle
```

三路 FSR：

```bash
python experiment_tools/python/collect_fsr.py --port /dev/cu.usbmodem101
```

單一 flex sensor 電阻值：

```bash
python experiment_tools/python/collect_flex_resistance.py --port /dev/cu.usbmodem101
```

只在終端機查看單一 flex ADC：

```bash
python experiment_tools/python/read_flex_adc.py --port /dev/cu.usbmodem101
```

相機測試：

```bash
python experiment_tools/python/camera_test.py --camera-index 0
```

同步記錄相機角度與 flex ADC：

```bash
python experiment_tools/python/record_flex_angle.py \
  --port /dev/cu.usbmodem101 \
  --camera-index 0
```

相機程式開始後用滑鼠框選 flex sensor 與兩塊黑膠帶。錄影期間：

- `C`：將目前角度校正為 0 度
- `R`：重新框選偵測區域
- `Q` 或 `Esc`：停止並儲存

所有產出預設放在執行位置的 `output/`，也可用 `--output-dir` 指定其他位置。

## Relay 注意事項

`relay_test.ino` 預設 relay 為 active HIGH：

```cpp
const uint8_t RELAY_ON = HIGH;
const uint8_t RELAY_OFF = LOW;
```

若模組為 active LOW，請把兩個值對調。上電前先確認 relay 模組、泵浦與電磁閥的額定電壓；Arduino
只負責低壓控制訊號，控制端需共地，高壓／110 V 電源側需保持隔離。

## 測試

不接硬體即可執行 parser 與角度計算測試：

```bash
python -m unittest discover -s experiment_tools/tests -v
```

Arduino 編譯檢查需要另行安裝 Arduino CLI 與對應 board core；實際接線前仍須確認板型、腳位與 relay
active level。

## 未打包內容

- CSV、MP4 與實驗資料集
- `.venv`、Python cache 與 IDE 設定
- 寫死的個人資料夾路徑與固定 USB port
- 舊檔的重複版本

整理來源包含原本的 `flex_sensor_data_collect.py`、`flex_data_single.py`、`read_flex.py`、
`record_camera_flex.py`、`camera_test.py`、Downloads 內的三個 data collector，以及 Documents/Arduino
內的 flex／resistance／relay sketches。
