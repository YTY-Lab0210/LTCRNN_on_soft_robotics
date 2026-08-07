# 上傳方式

這個資料夾是準備加入 `TWPHS/LTCRNN_on_soft_robotics` repository 的新增內容，不會覆蓋現有的
`arduino/`、`src/`、`data/` 等資料夾。

1. 把 `experiment_tools/` 整個資料夾複製到 repository 根目錄。
2. 在 repository 根目錄執行：

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r experiment_tools/requirements.txt
   python -m unittest discover -s experiment_tools/tests -v
   ```

3. 確認沒有 CSV、MP4、資料集或虛擬環境被加入：

   ```bash
   git status --short
   ```

4. 提交並推送：

   ```bash
   git add experiment_tools
   git commit -m "Add data collection and relay test tools"
   git push origin main
   ```

如果主分支禁止直接 push，請建立分支後開 Pull Request：

```bash
git switch -c add-experiment-tools
git push -u origin add-experiment-tools
```

