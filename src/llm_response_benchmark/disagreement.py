"""Evaluator-disagreement workflow: ambiguity flags, adjudication queue, annotator calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import Dataset
from .schemas import DIMENSIONS
from .textutils import shorten


@dataclass
class DisagreementConfig:
    range_threshold: float = 1.75  # max - min across (bias-corrected) annotator scores
    std_threshold: float = 0.8  # population std across annotators on a dimension
    min_flagged_dims: int = 2  # dims in dispute before an item counts as ambiguous
    severe_range: float = 2.75  # a single dimension this split is ambiguous on its own
    outlier_gap: float = 2.0  # |score - median(others)| for an outlier annotator
    bias_correct: bool = True  # subtract each annotator's per-dimension bias first
    lenient_threshold: float = 0.3
    harsh_threshold: float = -0.3
    compressed_ratio: float = 0.75
    noisy_corr: float = 0.6


# ---------------------------------------------------------------------------
# Annotator bias / consensus
# ---------------------------------------------------------------------------
def loo_consensus(m: np.ndarray) -> np.ndarray:
    """Leave-one-out consensus: for each (annotator, item), the mean of the *other* annotators."""
    present = ~np.isnan(m)
    total = np.nansum(m, axis=0)
    count = present.sum(axis=0)
    others_n = count[None, :] - present
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (total[None, :] - np.where(present, m, 0.0)) / others_n
    loo[others_n == 0] = np.nan
    return loo


def fit_additive_biases(m: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Fit s[a, i] = q[i] + b[a] by alternating least squares (missing values ignored).

    Biases are centred so that the median annotator has bias 0. Median centring keeps a single
    lenient (or harsh) annotator from dragging everyone else's bias estimate in the opposite
    direction, which is what happens with a plain leave-one-out comparison.
    """
    present = ~np.isnan(m)
    b = np.zeros(m.shape[0])
    for _ in range(n_iter):
        q = np.nanmean(m - b[:, None], axis=0)
        resid = np.where(present, m - q[None, :], np.nan)
        b_new = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else 0.0 for r in resid])
        b_new -= np.median(b_new)
        if np.allclose(b_new, b, atol=1e-9):
            b = b_new
            break
        b = b_new
    return b


def annotator_biases(ds: Dataset) -> dict[str, dict[str, float]]:
    """Per-annotator, per-dimension additive bias (see `fit_additive_biases`)."""
    out: dict[str, dict[str, float]] = {a: {} for a in ds.annotators}
    for dim in DIMENSIONS:
        m, raters, _ = ds.score_matrix(dim)
        b = fit_additive_biases(m)
        for i, a in enumerate(raters):
            out[a][dim] = float(b[i])
    return out


def consensus_scores(ds: Dataset, bias_correct: bool = False) -> dict[str, dict[str, float]]:
    """Per response and dimension, the mean annotator score (optionally bias-corrected, clipped to 1-5)."""
    biases = annotator_biases(ds) if bias_correct else None
    out: dict[str, dict[str, float]] = {}
    for dim in DIMENSIONS:
        m, raters, rids = ds.score_matrix(dim)
        if biases:
            m = m - np.array([biases[a][dim] for a in raters])[:, None]
        mean = np.clip(np.nanmean(m, axis=0), 1, 5)
        for j, rid in enumerate(rids):
            out.setdefault(rid, {})[dim] = float(mean[j])
    return out


# ---------------------------------------------------------------------------
# Item-level disagreement and ambiguity flags
# ---------------------------------------------------------------------------
def item_disagreement(ds: Dataset, cfg: DisagreementConfig | None = None) -> list[dict]:
    """One record per (response, dimension) with spread statistics and an outlier annotator if any."""
    cfg = cfg or DisagreementConfig()
    biases = annotator_biases(ds) if cfg.bias_correct else None
    records = []
    for dim in DIMENSIONS:
        m, raters, rids = ds.score_matrix(dim)
        adj = m - np.array([biases[a][dim] for a in raters])[:, None] if biases else m
        for j, rid in enumerate(rids):
            col, acol = m[:, j], adj[:, j]
            mask = ~np.isnan(col)
            if mask.sum() < 2:
                continue
            vals = acol[mask]
            names = [r for r, ok in zip(raters, mask) if ok]
            outlier = None
            for k, name in enumerate(names):
                others = np.delete(vals, k)
                if abs(vals[k] - np.median(others)) >= cfg.outlier_gap and np.ptp(others) <= 1:
                    outlier = name
            rng_ = float(np.ptp(vals))
            std = float(np.std(vals))
            records.append(
                {
                    "response_id": rid,
                    "dimension": dim,
                    "scores": {r: int(v) for r, v in zip(names, col[mask])},
                    "mean": float(np.mean(col[mask])),
                    "range": rng_,
                    "std": std,
                    "raw_range": float(np.ptp(col[mask])),
                    "outlier": outlier,
                    "disputed": bool(rng_ >= cfg.range_threshold - 1e-9 or std >= cfg.std_threshold),
                }
            )
    return records


def flag_items(ds: Dataset, cfg: DisagreementConfig | None = None) -> list[dict]:
    """Aggregate dimension-level disputes into response-level flags with a priority score.

    Labels:
      ambiguous          - >= min_flagged_dims disputed dimensions, or one dimension split by severe_range
      outlier_annotator  - disputes explained by a single annotator far from the other two
      dimension_dispute  - a single disputed dimension
    """
    cfg = cfg or DisagreementConfig()
    recs = item_disagreement(ds, cfg)
    by_resp: dict[str, list[dict]] = {}
    for r in recs:
        by_resp.setdefault(r["response_id"], []).append(r)
    rmap = ds.response_by_id
    flagged = []
    for rid, rows in by_resp.items():
        disputed = [r for r in rows if r["disputed"]]
        if not disputed:
            continue
        severe = any(r["range"] >= cfg.severe_range - 1e-9 for r in disputed)
        outliers = {r["outlier"] for r in disputed if r["outlier"]}
        if len(disputed) >= cfg.min_flagged_dims or severe:
            label = "ambiguous"
        elif outliers:
            label = "outlier_annotator"
        else:
            label = "dimension_dispute"
        mean_std = float(np.mean([r["std"] for r in rows]))
        priority = round(sum(r["range"] for r in disputed) + 2.0 * mean_std + (2.0 if label == "ambiguous" else 0.0), 3)
        flagged.append(
            {
                "response_id": rid,
                "prompt_id": rmap[rid].prompt_id,
                "model": rmap[rid].model,
                "label": label,
                "priority": priority,
                "mean_std": round(mean_std, 3),
                "n_disputed_dims": len(disputed),
                "disputed_dimensions": [r["dimension"] for r in disputed],
                "outlier_annotators": sorted(outliers),
            }
        )
    flagged.sort(key=lambda f: (-f["priority"], f["response_id"]))
    return flagged


def adjudication_queue(ds: Dataset, cfg: DisagreementConfig | None = None, include_labels=("ambiguous", "outlier_annotator", "dimension_dispute")) -> list[dict]:
    """Build adjudication tasks (one per flagged response), most contentious first."""
    cfg = cfg or DisagreementConfig()
    recs = {(r["response_id"], r["dimension"]): r for r in item_disagreement(ds, cfg)}
    rmap, pmap = ds.response_by_id, ds.prompt_by_id
    ann_by = {(a.response_id, a.annotator_id): a for a in ds.annotations}
    queue = []
    for rank, f in enumerate([f for f in flag_items(ds, cfg) if f["label"] in include_labels], start=1):
        resp = rmap[f["response_id"]]
        dims = []
        for dim in f["disputed_dimensions"]:
            r = recs[(f["response_id"], dim)]
            dims.append(
                {
                    "dimension": dim,
                    "scores": r["scores"],
                    "range": r["raw_range"],
                    "std": round(r["std"], 3),
                    "outlier": r["outlier"],
                    "rationales": {
                        a: ann_by[(f["response_id"], a)].rationales.get(dim, "") for a in r["scores"]
                    },
                }
            )
        queue.append(
            {
                "queue_rank": rank,
                "response_id": f["response_id"],
                "prompt_id": resp.prompt_id,
                "model": resp.model,
                "label": f["label"],
                "priority": f["priority"],
                "prompt": shorten(pmap[resp.prompt_id].text, 160),
                "response_excerpt": shorten(resp.text, 240),
                "disputed": dims,
                "status": "pending",
                "resolution": None,
            }
        )
    return queue


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(ds: Dataset, cfg: DisagreementConfig | None = None) -> dict:
    """Per-annotator bias, spread and agreement versus leave-one-out consensus and gold labels."""
    cfg = cfg or DisagreementConfig()
    gold = {g.response_id: g.scores for g in ds.gold}
    result: dict = {"annotators": {}, "thresholds": {
        "lenient_bias": cfg.lenient_threshold, "harsh_bias": cfg.harsh_threshold,
        "compressed_spread_ratio": cfg.compressed_ratio, "noisy_corr": cfg.noisy_corr}}
    per: dict[str, dict[str, dict]] = {a: {} for a in ds.annotators}
    for dim in DIMENSIONS:
        m, raters, rids = ds.score_matrix(dim)
        loo = loo_consensus(m)
        g = np.array([gold.get(r, {}).get(dim, np.nan) for r in rids], dtype=float)
        add_b = fit_additive_biases(m)
        for i, a in enumerate(raters):
            s, c = m[i], loo[i]
            mask = ~(np.isnan(s) | np.isnan(c))
            s_, c_ = s[mask], c[mask]
            sd_c = float(np.std(c_))
            corr = float(np.corrcoef(s_, c_)[0, 1]) if len(s_) > 2 and np.std(s_) > 0 and sd_c > 0 else float("nan")
            stats = {
                "n": int(mask.sum()),
                "mean_score": float(np.mean(s_)),
                "bias": float(add_b[i]),
                "loo_bias": float(np.mean(s_ - c_)),
                "mae": float(np.mean(np.abs(s_ - c_))),
                "spread_ratio": float(np.std(s_) / sd_c) if sd_c > 0 else float("nan"),
                "corr": corr,
            }
            gm = ~(np.isnan(s) | np.isnan(g))
            if gm.any():
                stats["gold_n"] = int(gm.sum())
                stats["gold_bias"] = float(np.mean(s[gm] - g[gm]))
                stats["gold_mae"] = float(np.mean(np.abs(s[gm] - g[gm])))
                stats["gold_exact"] = float(np.mean(s[gm] == g[gm]))
            per[a][dim] = stats

    for a, dims in per.items():
        def avg(key: str) -> float | None:
            vals = [d[key] for d in dims.values() if key in d and not np.isnan(d[key])]
            return float(np.mean(vals)) if vals else None

        overall = {k: avg(k) for k in ("mean_score", "bias", "loo_bias", "mae", "spread_ratio", "corr", "gold_bias", "gold_mae", "gold_exact")}
        flags = []
        if overall["bias"] is not None and overall["bias"] >= cfg.lenient_threshold:
            flags.append("lenient")
        if overall["bias"] is not None and overall["bias"] <= cfg.harsh_threshold:
            flags.append("harsh")
        if overall["spread_ratio"] is not None and overall["spread_ratio"] < cfg.compressed_ratio:
            flags.append("compressed_range")
        if overall["corr"] is not None and overall["corr"] < cfg.noisy_corr:
            flags.append("noisy")
        dim_flags = {}
        for dim, d in dims.items():
            f = []
            if d["bias"] >= cfg.lenient_threshold:
                f.append("lenient")
            if d["bias"] <= cfg.harsh_threshold:
                f.append("harsh")
            if not np.isnan(d["corr"]) and d["corr"] < cfg.noisy_corr:
                f.append("noisy")
            if f:
                dim_flags[dim] = f
        result["annotators"][a] = {
            "overall": overall,
            "flags": flags,
            "dimension_flags": dim_flags,
            "per_dimension": dims,
            "recommendation": _recommend(flags, overall),
        }
    return result


def _recommend(flags: list[str], overall: dict) -> str:
    if not flags:
        return "Calibrated; no action needed."
    tips = []
    if "lenient" in flags:
        tips.append(f"scores {overall['bias']:+.2f} above peers on average; review anchor descriptions for scores 2-3 and re-score gold items")
    if "harsh" in flags:
        tips.append(f"scores {overall['bias']:+.2f} below peers on average; review anchors for scores 4-5")
    if "compressed_range" in flags:
        tips.append("uses a narrow part of the scale; encourage using the full 1-5 range")
    if "noisy" in flags:
        tips.append("weak correlation with peers; schedule a calibration session on disputed items")
    return "; ".join(tips).capitalize() + "."
