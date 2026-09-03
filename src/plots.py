"""Plots for the virtual-screening workflow.

Kept out of the notebook because the matplotlib boilerplate is long and repetitive;
the notebook cells stay short enough to read at a glance.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .metrics import enrichment_curve

ACTIVE = "#c44e52"
INACTIVE = "#4c72b0"
ACCENT = "#dd8452"


def plot_cv_metrics(fold_metrics, exclude_prefix="Hits@"):
    """Bar chart of each metric across cross-validation folds, mean and spread."""
    names = [m for m in fold_metrics[0]
             if not m.startswith(exclude_prefix) and fold_metrics[0][m] is not None]
    means = [np.mean([f[m] for f in fold_metrics]) for m in names]
    stds = [np.std([f[m] for f in fold_metrics]) for m in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, means, yerr=stds, capsize=4, color=INACTIVE)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"Cross-validation metrics over {len(fold_metrics)} folds "
                 "(bars show the spread)", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    for x, (m, s) in enumerate(zip(means, stds)):
        ax.text(x, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_score_distribution(y_true, y_score, bins=50, title="Predicted score by true class"):
    """Where the actives sit in the score distribution, against the inactive bulk.

    The inactive count is on a log scale - with 0.18% actives a linear axis would
    render the active bars invisible.
    """
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    edges = np.linspace(0, 1, bins + 1)
    ax.hist(s[y == 0], bins=edges, color=INACTIVE, alpha=0.85,
            label=f"inactive (n={int((y == 0).sum())})")
    ax.hist(s[y == 1], bins=edges, color=ACTIVE, alpha=0.95,
            label=f"active (n={int((y == 1).sum())})")
    ax.set_yscale("log")
    ax.set_xlabel("predicted probability of being active")
    ax.set_ylabel("compounds (log scale)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr_and_roc(y_true, y_score):
    """Precision-recall beside ROC. On rare positives, trust the PR curve."""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    base = y.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    precision, recall, _ = precision_recall_curve(y, s)
    axes[0].plot(recall, precision, color=ACTIVE, linewidth=2)
    axes[0].axhline(base, color="grey", linestyle="--", linewidth=1,
                    label=f"random = {base:.3%}")
    axes[0].set_xlabel("recall (fraction of actives found)")
    axes[0].set_ylabel("precision (fraction of picks that are real)")
    axes[0].set_title(f"Precision-recall   AP = {average_precision_score(y, s):.3f}",
                      fontweight="bold")
    axes[0].legend()

    fpr, tpr, _ = roc_curve(y, s)
    axes[1].plot(fpr, tpr, color=INACTIVE, linewidth=2)
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="random")
    axes[1].set_xlabel("false positive rate")
    axes[1].set_ylabel("true positive rate")
    axes[1].set_title(f"ROC   AUC = {roc_auc_score(y, s):.3f}", fontweight="bold")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_enrichment_curve(y_true, y_score, zoom_frac=0.1):
    """Actives recovered against compounds screened, with the random diagonal.

    The right-hand panel zooms into the shallow end, which is the only part you
    can usually afford to assay.
    """
    screened, found = enrichment_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, limit in zip(axes, [1.0, zoom_frac]):
        ax.plot(screened, found, color=ACTIVE, linewidth=2, label="model")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="random")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("fraction of the library screened")
        ax.set_ylabel("fraction of actives found")
        ax.legend(loc="lower right")
    axes[0].set_title("Enrichment curve", fontweight="bold")
    axes[1].set_title(f"Zoomed to the top {zoom_frac:.0%}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(thresholds, n_selected, actives_kept, total_actives):
    """How many compounds you would assay, and how many hits you would keep."""
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(thresholds, n_selected, marker="o", color=INACTIVE, linewidth=2)
    ax1.set_xlabel("Final_Score threshold")
    ax1.set_ylabel("compounds nominated", color=INACTIVE)
    ax1.tick_params(axis="y", labelcolor=INACTIVE)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, actives_kept, marker="s", color=ACTIVE, linewidth=2)
    ax2.set_ylabel(f"actives kept (of {total_actives})", color=ACTIVE)
    ax2.set_ylim(0, total_actives + 0.5)
    ax2.tick_params(axis="y", labelcolor=ACTIVE)

    ax1.set_title("A stricter threshold costs you real hits", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_molecule_grid(smiles, legends=None, n=12, mols_per_row=4, size=(260, 220)):
    """Draw the top compounds as actual structures.

    Returns an RDKit grid image, which Jupyter and Colab render inline.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    smiles = list(smiles)[:n]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in keep]
    if legends is not None:
        legends = [list(legends)[:n][i] for i in keep]

    return Draw.MolsToGridImage(mols, molsPerRow=mols_per_row,
                                subImgSize=size, legends=legends)
