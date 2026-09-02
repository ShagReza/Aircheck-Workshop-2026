# Aircheck Workshop 2026

Materials for the Aircheck Workshop 2026.

## Repository layout

- `notebooks/` contains the workshop notebooks.
- `src/` contains reusable Python code.
- `data/` contains the workshop datasets (see below). Full datasets remain in cloud storage.
- `results/` contains generated outputs and model artifacts.
- `requirements.txt` lists the Python dependencies used by the notebooks.

## Workshop data

The notebooks read these files from `data/`. They are committed to the repository so every
participant starts from identical data, in Colab and locally alike.

| File | Compounds | Label | SMILES | What it is |
|---|---|---|---|---|
| `sample-train.parquet` | 4,000 | yes, balanced 50/50 | no | DEL screen against WDR91. Training set. |
| `sample-test.parquet` | 2,000 | yes, balanced 50/50 | no | Held-out slice of the DEL screen. Quick sanity check. |
| `sample-test-2.parquet` | 5,000 | yes, 9 actives (0.18%) | yes | Realistic evaluation set. |
| `sample-screen.parquet` | 5,000 | no | yes | Compounds to screen and nominate. |

Every fingerprint column (`ECFP4`, `ECFP6`, `FCFP4`, `FCFP6`, `MACCS`, `RDK`, `AVALON`,
`ATOMPAIR`, `TOPTOR`) is stored as an **array of counts**, so reading a column gives NumPy
arrays directly - there are no comma-separated strings to parse.

The full AIRCHECK datasets are available from https://www.aircheck.ai/datasets

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

## How the notebooks find their data

The two hands-on notebooks open with a bootstrap cell that works in both environments:

- **In Colab** it clones this repository into the runtime, then installs everything in
  `requirements.txt`.
- **Locally** it walks up from the notebook to find the repository root and installs
  nothing, assuming you already set up a virtual environment.

Either way the notebook ends up with `REPO_ROOT`, `DATA_DIR` and `RESULTS_DIR`, and every
later cell uses those instead of absolute paths. Models and outputs are written to
`results/`, which is gitignored.
