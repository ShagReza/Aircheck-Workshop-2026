# Aircheck Workshop 2026

Materials for the Aircheck Workshop 2026.

## Repository layout

- `notebooks/` contains the workshop notebooks.
- `src/` contains reusable Python code.
- `data/` contains small, non-sensitive example data. Large datasets should remain in cloud storage.
- `results/` contains generated outputs and model artifacts.
- `requirements.txt` lists the Python dependencies used by the notebooks.

## Local setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Open the notebooks in VS Code with the Jupyter extension. Keep file paths relative to the repository so the notebooks can also be cloned into a Colab runtime later.

## Colab

Clone the repository into the Colab runtime, change into the repository directory, install `requirements.txt`, and open a notebook from `notebooks/`. Small example data committed under `data/` is available automatically; full datasets can be loaded from GCP or Azure storage without changing the analysis workflow.