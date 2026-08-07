# Upload Instructions

This folder was prepared as an add-on for the `TWPHS/LTCRNN_on_soft_robotics` repository. It does not overwrite the existing `arduino/`, `src/`, or `data/` folders.

1. Copy the full `experiment_tools/` folder into the repository root.
2. Run the following commands from the repository root:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r experiment_tools/requirements.txt
   python -m unittest discover -s experiment_tools/tests -v
   ```

3. Confirm that no CSV files, MP4 files, datasets, or virtual environments are staged:

   ```bash
   git status --short
   ```

4. Commit and push:

   ```bash
   git add experiment_tools
   git commit -m "Add data collection and relay test tools"
   git push origin main
   ```

If direct pushes to the main branch are blocked, create a branch and open a pull request:

```bash
git switch -c add-experiment-tools
git push -u origin add-experiment-tools
```
