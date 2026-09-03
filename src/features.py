"""Feature-matrix helpers: stacking fingerprints, fusing them, and rebalancing.

Everything here returns **new** arrays or new DataFrames. Nothing modifies the frame
you pass in, so the optional experiments in the notebook cannot disturb the main
training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stack_fingerprint(df, column):
    """One fingerprint column -> a (n_molecules, n_bits) float32 matrix."""
    return np.stack(df[column].to_numpy()).astype(np.float32)


def fuse_fingerprints(df, columns):
    """Lay several fingerprint columns side by side into one wider matrix.

    Each fingerprint describes the same molecule in a different way, so joining them
    gives the model more to work with. The cost is width: the matrix gets as wide as
    the sum of its parts, and training time grows with it.
    """
    if isinstance(columns, str):
        columns = [columns]
    return np.hstack([stack_fingerprint(df, c) for c in columns])


def fingerprint_widths(df, columns):
    """How many bits each fingerprint column contributes."""
    return {c: int(stack_fingerprint(df, c).shape[1]) for c in columns}


def fusion_summary(df, combinations):
    """Table of shape and relative cost for each combination of fingerprints."""
    rows = []
    base = None
    for combo in combinations:
        combo = [combo] if isinstance(combo, str) else list(combo)
        width = sum(fingerprint_widths(df, combo).values())
        base = width if base is None else base
        rows.append({
            "fingerprints": " + ".join(combo),
            "columns": len(combo),
            "features": width,
            "vs first row": f"{width / base:.2f}x",
        })
    return pd.DataFrame(rows)


def resample_negatives(df, n_negatives=None, ratio=None, label_col="LABEL",
                       random_state=42):
    """Return a COPY of `df` with the inactive rows subsampled.

    Give either `n_negatives` (an absolute count) or `ratio` (negatives per positive).
    Positives are always kept in full - they are the scarce, expensive part. The input
    frame is never modified.
    """
    positives = df[df[label_col] == 1]
    negatives = df[df[label_col] == 0]

    if ratio is not None:
        n_negatives = int(round(ratio * len(positives)))
    if n_negatives is None:
        raise ValueError("pass either n_negatives or ratio")

    n_negatives = min(int(n_negatives), len(negatives))
    kept = negatives.sample(n=n_negatives, random_state=random_state)
    out = pd.concat([positives, kept]).sample(frac=1.0, random_state=random_state)
    return out.reset_index(drop=True)


def balance_summary(df, negative_counts, label_col="LABEL", random_state=42):
    """Table of what each choice of negative count does to the training set."""
    rows = []
    for n in negative_counts:
        sub = resample_negatives(df, n_negatives=n, label_col=label_col,
                                 random_state=random_state)
        pos = int(sub[label_col].sum())
        neg = len(sub) - pos
        rows.append({
            "negatives kept": neg,
            "positives": pos,
            "total rows": len(sub),
            "negatives per positive": round(neg / pos, 2) if pos else float("nan"),
            "% active": f"{pos / len(sub):.1%}",
        })
    return pd.DataFrame(rows)
