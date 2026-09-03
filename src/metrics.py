"""Metrics for ranked virtual-screening results.

A screening model is not really a classifier - it is a *ranker*. You will only ever
assay the top slice of its output, so what matters is how many real actives sit in
the compounds you can afford to test. These helpers answer that directly.
"""
from __future__ import annotations

import numpy as np

DEFAULT_KS = (20, 50, 100, 200, 500)


def _rank(y_true, y_score):
    """Return the true labels reordered from highest predicted score to lowest."""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"labels {y.shape} and scores {s.shape} have different lengths")
    return y[np.argsort(-s, kind="stable")]


def hits_at_k(y_true, y_score, ks=DEFAULT_KS):
    """How many actives appear in the top-K highest scoring compounds.

    Values of K larger than the dataset are skipped, so this stays sensible on a
    small cross-validation fold as well as on a full screening library.
    """
    ranked = _rank(y_true, y_score)
    return {int(k): int(ranked[:k].sum()) for k in ks if k <= ranked.size}


def enrichment_at_k(y_true, y_score, ks=DEFAULT_KS):
    """Enrichment factor: the hit rate in the top K divided by the overall hit rate.

    An enrichment of 1.0 means the model did no better than picking at random.
    """
    ranked = _rank(y_true, y_score)
    base = ranked.mean()
    if base == 0:
        return {int(k): float("nan") for k in ks if k <= ranked.size}
    return {int(k): (int(ranked[:k].sum()) / k) / base
            for k in ks if k <= ranked.size}


def screening_table(y_true, y_score, ks=DEFAULT_KS):
    """A printable summary of hits, hit rate and enrichment at each K."""
    ranked = _rank(y_true, y_score)
    total_actives = int(ranked.sum())
    base = ranked.mean()

    out = [f"{'K':>6} {'hits':>6} {'of':>4} {'hit rate':>10} {'enrichment':>12}",
           f"{'-' * 6} {'-' * 6} {'-' * 4} {'-' * 10} {'-' * 12}"]
    for k in ks:
        if k > ranked.size:
            continue
        hits = int(ranked[:k].sum())
        rate = hits / k
        enrich = rate / base if base > 0 else float("nan")
        out.append(f"{k:>6} {hits:>6} {total_actives:>4} {rate:>9.1%} {enrich:>11.1f}x")
    out.append(f"\nbaseline hit rate: {base:.2%}  "
               f"({total_actives} actives in {ranked.size} compounds)")
    return "\n".join(out)


def enrichment_curve(y_true, y_score):
    """Cumulative actives found as you work down the ranked list.

    Returns (fraction_screened, fraction_of_actives_found), both starting at 0,
    which is exactly what you need to plot an enrichment curve.
    """
    ranked = _rank(y_true, y_score)
    total = ranked.sum()
    if total == 0:
        raise ValueError("no actives in y_true - an enrichment curve is undefined")
    found = np.concatenate([[0], np.cumsum(ranked)]) / total
    screened = np.arange(ranked.size + 1) / ranked.size
    return screened, found
