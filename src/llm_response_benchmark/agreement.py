"""Inter-annotator agreement metrics implemented with numpy.

Conventions: a ratings matrix has shape (n_raters, n_units); missing ratings are NaN.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np

Metric = Literal["nominal", "ordinal", "interval"]


# --------------------------------------------------------------------------
# Krippendorff's alpha
# --------------------------------------------------------------------------
def coincidence_matrix(ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build Krippendorff's coincidence matrix.

    Returns (values, o) where `values` are the sorted distinct observed values and
    o[c, k] is the number of c-k pairs within units, each unit weighted by 1/(m_u - 1).
    Units with fewer than two ratings are not pairable and are ignored.
    """
    ratings = np.asarray(ratings, dtype=float)
    values = np.unique(ratings[~np.isnan(ratings)])
    index = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for unit in ratings.T:
        vals = unit[~np.isnan(unit)]
        m = len(vals)
        if m < 2:
            continue
        counts = np.zeros(len(values))
        for v in vals:
            counts[index[v]] += 1
        o += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)
    return values, o


def _delta_squared(values: np.ndarray, n_c: np.ndarray, metric: Metric) -> np.ndarray:
    k = len(values)
    d = np.zeros((k, k))
    if metric == "nominal":
        d = 1.0 - np.eye(k)
    elif metric == "interval":
        d = (values[:, None] - values[None, :]) ** 2
    elif metric == "ordinal":
        cum = np.cumsum(n_c)
        for c in range(k):
            for j in range(k):
                lo, hi = min(c, j), max(c, j)
                between = cum[hi] - (cum[lo - 1] if lo > 0 else 0.0)
                d[c, j] = (between - (n_c[c] + n_c[j]) / 2.0) ** 2
    else:
        raise ValueError(f"unknown metric {metric}")
    return d


def krippendorff_alpha(ratings: np.ndarray, metric: Metric = "ordinal") -> float:
    """Krippendorff's alpha for a (raters x units) matrix with NaN for missing values.

    alpha = 1 - D_o / D_e with observed disagreement D_o = sum(o_ck * d_ck) / n and
    expected disagreement D_e = sum(n_c * n_k * d_ck) / (n * (n - 1)).
    Returns NaN when there is no variation at all (D_e == 0).
    """
    values, o = coincidence_matrix(ratings)
    if len(values) == 0:
        return float("nan")
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    d = _delta_squared(values, n_c, metric)
    d_o = (o * d).sum() / n
    d_e = (np.outer(n_c, n_c) * d).sum() / (n * (n - 1))
    if d_e == 0:
        return float("nan")
    return float(1.0 - d_o / d_e)


# --------------------------------------------------------------------------
# Cohen's kappa (two raters)
# --------------------------------------------------------------------------
def cohen_kappa(
    a: np.ndarray,
    b: np.ndarray,
    categories: list | np.ndarray | None = None,
    weights: Literal[None, "linear", "quadratic"] = None,
) -> float:
    """Cohen's kappa between two raters, optionally weighted.

    kappa = 1 - sum(w * O) / sum(w * E), where O is the observed joint proportion table,
    E the outer product of the marginals and w the disagreement weights (0 on the diagonal).
    Pairs where either rating is NaN are dropped.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return float("nan")
    cats = np.asarray(sorted(set(a) | set(b)) if categories is None else categories, dtype=float)
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights}")
    denom = (w * exp).sum()
    if denom == 0:
        # both raters used a single identical category: perfect but uninformative
        return 1.0 if (w * obs).sum() == 0 else float("nan")
    return float(1.0 - (w * obs).sum() / denom)


# --------------------------------------------------------------------------
# Fleiss' kappa (many raters, fixed number per unit)
# --------------------------------------------------------------------------
def fleiss_kappa_from_counts(counts: np.ndarray) -> float:
    """Fleiss' kappa from an (n_units x n_categories) table of rating counts.

    Every row must sum to the same number of raters n.
    """
    counts = np.asarray(counts, dtype=float)
    n_units = counts.shape[0]
    n = counts.sum(axis=1)
    if n_units == 0:
        return float("nan")
    if not np.allclose(n, n[0]):
        raise ValueError("Fleiss' kappa needs the same number of ratings per unit")
    n = n[0]
    if n < 2:
        return float("nan")
    p_j = counts.sum(axis=0) / (n_units * n)
    p_i = ((counts**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_e = (p_j**2).sum()
    if p_e == 1:
        return float("nan")
    return float((p_bar - p_e) / (1 - p_e))


def fleiss_kappa(ratings: np.ndarray, categories: list | np.ndarray | None = None) -> float:
    """Fleiss' kappa from a (raters x units) matrix; units with any missing rating are dropped."""
    ratings = np.asarray(ratings, dtype=float)
    complete = ratings[:, ~np.isnan(ratings).any(axis=0)]
    if complete.size == 0:
        return float("nan")
    cats = np.unique(complete) if categories is None else np.asarray(categories, dtype=float)
    counts = np.stack([(complete == c).sum(axis=0) for c in cats], axis=1)
    return fleiss_kappa_from_counts(counts)


# --------------------------------------------------------------------------
# Pairwise and summary helpers
# --------------------------------------------------------------------------
def pairwise_agreement(
    ratings: np.ndarray, raters: list[str], categories: list | None = None
) -> list[dict]:
    """Per pair of raters: exact and within-1 agreement, mean absolute difference, kappas."""
    out = []
    for (i, ra), (j, rb) in combinations(enumerate(raters), 2):
        a, b = ratings[i], ratings[j]
        mask = ~(np.isnan(a) | np.isnan(b))
        n = int(mask.sum())
        if n == 0:
            continue
        diff = np.abs(a[mask] - b[mask])
        out.append(
            {
                "rater_a": ra,
                "rater_b": rb,
                "n": n,
                "exact": float((diff == 0).mean()),
                "within_1": float((diff <= 1).mean()),
                "mean_abs_diff": float(diff.mean()),
                "mean_signed_diff": float((a[mask] - b[mask]).mean()),
                "cohen_kappa": cohen_kappa(a, b, categories),
                "cohen_kappa_quadratic": cohen_kappa(a, b, categories, weights="quadratic"),
            }
        )
    return out


def agreement_report(dataset, categories: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """Compute all agreement statistics for every rubric dimension of a Dataset."""
    from .disagreement import fit_additive_biases
    from .schemas import DIMENSIONS

    cats = list(categories)
    per_dim = {}
    for dim in DIMENSIONS:
        m, raters, units = dataset.score_matrix(dim)
        pairs = pairwise_agreement(m, raters, cats)
        corrected = m - fit_additive_biases(m)[:, None]
        per_dim[dim] = {
            "n_units": len(units),
            "krippendorff_alpha_ordinal": krippendorff_alpha(m, "ordinal"),
            "krippendorff_alpha_interval": krippendorff_alpha(m, "interval"),
            "krippendorff_alpha_nominal": krippendorff_alpha(m, "nominal"),
            # interval alpha after removing each annotator's additive offset: how much agreement
            # would improve if systematic leniency/harshness were calibrated away
            "krippendorff_alpha_interval_bias_corrected": krippendorff_alpha(corrected, "interval"),
            "fleiss_kappa": fleiss_kappa(m, cats),
            "mean_cohen_kappa": float(np.nanmean([p["cohen_kappa"] for p in pairs])) if pairs else None,
            "mean_cohen_kappa_quadratic": float(np.nanmean([p["cohen_kappa_quadratic"] for p in pairs]))
            if pairs
            else None,
            "pairwise": pairs,
        }
    return {"raters": dataset.annotators, "dimensions": per_dim}


def interpret_alpha(alpha: float | None) -> str:
    """Krippendorff's rough guideline: >= .800 reliable, .667-.800 tentative, below that weak."""
    if alpha is None or np.isnan(alpha):
        return "undefined"
    if alpha >= 0.8:
        return "reliable"
    if alpha >= 0.667:
        return "tentative"
    return "weak"
