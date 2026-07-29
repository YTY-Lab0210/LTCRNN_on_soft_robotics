# Paper Figures 程式說明

此資料夾保存論文與簡報用圖表、重畫圖腳本、精簡結果表與輸出圖片。

## 主要重畫圖腳本

| 檔案 | 用途 |
| --- | --- |
| `make_paper_compact_figures.py` | 產生 compact benchmark 圖與無數字 confusion matrix |
| `redraw_mean_confusion_full_labels.py` | 產生完整類別標籤版本 confusion matrix |
| `make_accuracy_f1_bar_bptt_channels.py` | 產生 3-channel / 5-channel Accuracy 與 Macro-F1 長條圖 |
| `make_extended_benchmark_accuracy_by_family.py` | 將 Vanilla RNN、LSTM、LTC 按模型家族整理成 benchmark 圖 |
| `make_ltc_neuron_sweep_5ch_3ch_bar.py` | 產生 LTC-1/2/4/8/16 的 5-channel / 3-channel 長條圖 |
| `make_ltc_neuron_sweep_5ch_3ch_line.py` | 產生 LTC neuron sweep 折線圖 |
| `make_few_shot_models_5ch_3ch.py` | 產生 few-shot 訓練樣本數曲線圖 |
| `make_model_parameter_bar.py` | 產生模型參數量比較圖 |
| `make_ltc_parameter_sweep_bar.py` | 產生 LTC 不同神經元參數量比較圖 |

## 圖片與示意圖腳本

| 檔案 | 用途 |
| --- | --- |
| `make_single_finger_waveform.py` | 產生單指抓握 ADC 波形圖 |
| `remove_flex_sensor_background.py` | flex sensor 圖片去背 |
| `remove_soft_finger_background.py` | 軟式手指圖片去背 |

## 資料夾

| 資料夾 | 內容 |
| --- | --- |
| `source_tables/` | 供 figure scripts 使用的精簡 BPTT 結果表 |
| `extended_benchmark/` | extended benchmark 圖與 summary CSV |
| `few_shot_10_60/` | few-shot 10 到 60 samples per object 的圖與 CSV |
| `extended_benchmark_smoke/` | smoke test 或短版 benchmark 輸出 |

## 常用輸出圖

| 圖片 | 用途 |
| --- | --- |
| `paper_benchmark_accuracy_f1_compact.png` | paper-style 模型 benchmark |
| `paper_confusion_no_numbers_5ch_3ch_ids.png` | 小尺寸 confusion matrix，用 class id 顯示 |
| `ltc_neuron_sweep_5ch_3ch_bar.png` | LTC neuron number sweep |
| `model_parameter_count_bar.png` | 模型參數量比較 |
