# Dataset 說明

本資料夾保存重新整理後的 raw ADC dataset。資料來自五指仿人軟式機械手，每根手指貼附 flex sensor，量測抓握過程中的彎曲變化。

## 5-channel dataset

主要資料集：

```text
dataset_new_new_new/
```

每個 CSV 檔為一筆抓握序列，欄位如下：

```text
Time_ms, Thumb, Index, Middle, Ring, Pinky
```

每筆資料包含 400 個取樣點，取樣間隔為 10 ms。加上 header 後，每個 CSV 檔共 401 行。

## 3-channel dataset

3-channel 資料集：

```text
dataset_new_new_new_3ch/
```

保留欄位：

```text
Time_ms, Thumb, Middle, Pinky
```

此版本由 5-channel dataset 擷取指定手指後產生，並在每個類別內隨機重新編號。來源對照表：

```text
dataset_new_new_new_3ch_manifest.csv
```

## 檔名規則

檔名使用 confusion matrix 中的顯示標籤：

```text
Baseball_001.csv
Bottle_001.csv
Sponge Dice_001.csv
3D-Printed Part_001.csv
```

## 類別與數量

```text
3D-Printed Part   100
Baseball          100
Bottle            100
Optical Mouse     100
Plush Toy         100
Rubik's Cube      100
Smartphone        100
Sponge Dice       100
Stuffed Ball      100
Tape              100
```

總數：

```text
1000 CSV files
```

## 前處理

此資料夾保存 raw ADC 數值，沒有直接覆蓋成 normalized data。訓練時可由 training set 計算 mean 與 standard deviation，再用 z-score normalization 套用到 train / validation / test。

相關工具：

```text
src/utils/z_score.py
src/utils/create_3ch_dataset.py
src/utils/split_data.py
```
