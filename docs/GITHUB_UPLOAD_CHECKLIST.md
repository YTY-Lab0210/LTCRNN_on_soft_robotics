# GitHub Upload Checklist

Before pushing this repository publicly, check the following items:

- The working tree contains no private thesis PDFs, approval forms, authorization forms, or internal draft files.
- Generated caches such as `__pycache__/`, `.venv/`, `.npy`, `.npz`, model checkpoints, and large intermediate outputs are excluded.
- Dataset folders contain only the intended cleaned CSV files.
- README files describe the project, dataset format, experiment scripts, and Arduino deployment flow.
- Arduino sketches use the correct relay active level for the hardware test setup.
- Figure scripts point to committed source tables or documented local inputs.
- Run parser tests before committing:

```bash
python -m unittest discover -s experiment_tools/tests -v
```
