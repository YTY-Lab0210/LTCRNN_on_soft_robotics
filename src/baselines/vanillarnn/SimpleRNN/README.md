# Vanilla RNN / SimpleRNN Scripts

此資料夾保存早期 Vanilla RNN baseline 訓練程式。檔名仍使用 `SimpleRNN`，但在論文與圖表呈現時建議稱為 `Vanilla RNN`。

| 檔案 | 用途 |
| --- | --- |
| `SimpleRNN.py` | 訓練固定 hidden units 的 Vanilla RNN，輸入為 400-step flex sensor sequence |
| `SimpleRNN_different_neuron.py` | 測試不同 RNN hidden units 對分類表現的影響 |
| `SimpleRNN_reduce_trainingset.py` | 測試減少 training samples 時 Vanilla RNN 的表現 |

## 輸入資料

早期腳本預期資料已分成：

```text
training/
validation/
test/
```

若要用目前 repo 內的新 dataset 重跑，需先更新程式內的資料路徑與前處理方式。
