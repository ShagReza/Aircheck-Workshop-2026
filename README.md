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

## Open in Colab

The notebooks run in Google Colab with no setup — click a link below and Colab opens the
notebook straight from this repository. The repository is public, so participants do not
need a GitHub account.

| Notebook | Open |
|---|---|
| Introduction to Python | [Open in Colab](https://colab.research.google.com/github/ShagReza/Aircheck-Workshop-2026/blob/main/notebooks/Introduction-Python-AircheckWorkshop2026.ipynb) |
| Introduction to Machine Learning | [Open in Colab](https://colab.research.google.com/github/ShagReza/Aircheck-Workshop-2026/blob/main/notebooks/Introduction-MachineLearning-AircheckWorkshop2026.ipynb) |
| Hands-On: Data Exploration | [Open in Colab](https://colab.research.google.com/github/ShagReza/Aircheck-Workshop-2026/blob/main/notebooks/HandsOn-DataExploration-AircheckWorkshop2026.ipynb) |
| Hands-On: Machine Learning | [Open in Colab](https://colab.research.google.com/github/ShagReza/Aircheck-Workshop-2026/blob/main/notebooks/HandsOn-MachineLearning-AircheckWorkshop2026.ipynb) |

### Notes for participants

- Colab opens a **copy** of the notebook. Your edits are not saved back to this repository.
  To keep your work, use `File -> Save a copy in Drive` before you start editing.
- If the notebook is updated during the workshop, re-open the link above to get the new
  version. Your saved Drive copy will not update on its own.
- Colab provides `pandas`, `numpy` and `scikit-learn` by default; the hands-on notebooks install the
  remaining packages (`rdkit`, `lightgbm`, `umap-learn`) in their first cell.

### Other ways to open a notebook in Colab

- **Rewrite the URL.** Any notebook in a public GitHub repository can be opened by putting
  `colab.research.google.com/github/` in front of the `owner/repo/blob/...` part of its
  GitHub URL.
- **From inside Colab.** `File -> Open notebook -> GitHub`, then enter
  `ShagReza/Aircheck-Workshop-2026` and pick a notebook from the list.

## Running the notebooks against the full datasets

Clone the repository into the Colab runtime, change into the repository directory, install
`requirements.txt`, and open a notebook from `notebooks/`. Small example data committed
under `data/` is available automatically; full datasets can be loaded from GCP or Azure
storage without changing the analysis workflow.
