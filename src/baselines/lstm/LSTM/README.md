# LSTM Scripts

此資料夾保存早期 LSTM baseline 訓練程式。

| 檔案 | 用途 |
| --- | --- |
| `lstm.py` | 訓練固定 hidden units 的 LSTM baseline |
| `lstm_different_neuron.py` | 測試不同 LSTM hidden units 對分類表現的影響 |
| `lstm_reduce_trainingset.py` | 測試減少 training samples 時 LSTM 的表現 |

## 輸入資料

早期腳本預期資料已分成：

```text
training/
validation/
test/
```

若要用目前 repo 內的新 dataset 重跑，需先更新資料路徑，或改用主 benchmark 腳本。
