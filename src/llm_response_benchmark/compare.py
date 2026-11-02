"""Model comparison: per-dimension means with bootstrap CIs, pairwise win rates."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .dataset import Dataset
from .disagreement import consensus_scores
from .schemas import DIMENSIONS


def bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of `x`. Returns (mean, low, high)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(x.mean()), float(lo), float(hi)


def score_tensor(
    ds: Dataset, bias_correct: bool = False, weights: dict[str, float] | None = None
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Return (T[model, prompt, dim+1], models, prompt_ids, dims) where the last slot is the
    weighted overall score. Only prompts answered by every model are kept (paired design)."""
    cons = consensus_scores(ds, bias_correct=bias_correct)
    models = ds.models
    by_pm = {(r.prompt_id, r.model): r.response_id for r in ds.responses if r.response_id in cons}
    prompt_ids = [p.prompt_id for p in ds.prompts if all((p.prompt_id, m) in by_pm for m in models)]
    dims = list(DIMENSIONS)
    w = np.array([(weights or {}).get(d, 1.0) for d in dims])
    w = w / w.sum()
    t = np.zeros((len(models), len(prompt_ids), len(dims) + 1))
    for i, m in enumerate(models):
        for j, p in enumerate(prompt_ids):
            s = cons[by_pm[(p, m)]]
            row = np.array([s[d] for d in dims])
            t[i, j, :-1] = row
            t[i, j, -1] = float(row @ w)
    return t, models, prompt_ids, dims + ["overall"]


def compare_models(
    ds: Dataset,
    n_boot: int = 2000,
    seed: int = 0,
    bias_correct: bool = False,
    tie_margin: float = 0.25,
    weights: dict[str, float] | None = None,
) -> dict:
    t, models, prompt_ids, dims = score_tensor(ds, bias_correct, weights)
    n = len(prompt_ids)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))  # shared resamples => paired comparisons
    lo_q, hi_q = 0.025, 0.975

    per_dim: dict[str, dict[str, dict]] = {}
    for k, d in enumerate(dims):
        per_dim[d] = {}
        for i, m in enumerate(models):
            x = t[i, :, k]
            means = x[boot_idx].mean(axis=1)
            per_dim[d][m] = {
                "mean": float(x.mean()),
                "ci_low": float(np.quantile(means, lo_q)),
                "ci_high": float(np.quantile(means, hi_q)),
            }

    wins = []
    for (i, a), (j, b) in combinations(enumerate(models), 2):
        for k, d in enumerate(dims):
            diff = t[i, :, k] - t[j, :, k]
            w_ = int((diff > tie_margin).sum())
            l_ = int((diff < -tie_margin).sum())
            ties = n - w_ - l_
            bd = diff[boot_idx].mean(axis=1)
            lo, hi = float(np.quantile(bd, lo_q)), float(np.quantile(bd, hi_q))
            wins.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "dimension": d,
                    "wins": w_,
                    "losses": l_,
                    "ties": ties,
                    "win_rate": (w_ + 0.5 * ties) / n if n else float("nan"),
                    "mean_diff": float(diff.mean()),
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "significant": bool(lo > 0 or hi < 0),
                }
            )

    cat_of = {p.prompt_id: p.category for p in ds.prompts}
    cats = sorted({cat_of[p] for p in prompt_ids})
    by_cat = {
        c: {m: float(t[i, [j for j, p in enumerate(prompt_ids) if cat_of[p] == c], -1].mean()) for i, m in enumerate(models)}
        for c in cats
    }
    ranking = sorted(models, key=lambda m: -per_dim["overall"][m]["mean"])
    return {
        "models": models,
        "n_prompts": n,
        "n_boot": n_boot,
        "bias_corrected": bias_correct,
        "tie_margin": tie_margin,
        "dimensions": per_dim,
        "win_rates": wins,
        "by_category": by_cat,
        "ranking": ranking,
    }


def win_matrix(comparison: dict, dimension: str = "overall") -> dict[str, dict[str, float]]:
    """Square matrix view: win_matrix[a][b] = share of prompts where a beats b (ties count half)."""
    models = comparison["models"]
    mat = {a: {b: float("nan") for b in models} for a in models}
    for w in comparison["win_rates"]:
        if w["dimension"] != dimension:
            continue
        mat[w["model_a"]][w["model_b"]] = w["win_rate"]
        mat[w["model_b"]][w["model_a"]] = 1.0 - w["win_rate"]
    return mat
