# 未納入 Repository 的檔案

原始工作資料夾包含許多不適合放入公開 GitHub repository 的內容，因此本 release folder 只保留可重現實驗、繪圖與部署所需的核心資料。

未納入類別：

```text
_unused_datasets_backup_20260717/   舊資料集備份
.codex_tmp/                         暫存與自動產生檔案
*.npy                               本機訓練 cache
__pycache__/                        Python bytecode cache
private thesis/front-matter PDFs    審定書、推薦書、授權書等簽核文件
teacher paper draft files           老師 paper draft 與內部討論文件
large generated output folders      大型中間輸出與重複實驗資料夾
```

此 repository 的目標是讓使用者能理解資料格式、重跑主要實驗、重畫論文圖表，並測試 Arduino deployment；不保存私有文件與過大的中間檔。
