"""Structured per-model feedback built from consensus scores, comparisons and annotator rationales."""

from __future__ import annotations

import numpy as np

from .compare import win_matrix
from .dataset import Dataset
from .disagreement import consensus_scores
from .schemas import DIMENSIONS
from .textutils import shorten

ADVICE = {
    "factual_accuracy": "Audit the incorrect claims below; consider retrieval grounding or answer verification for factual prompts.",
    "relevance": "Trim tangents and padding; responses drift away from the question.",
    "completeness": "Responses omit reference key points; prompt the model to cover all parts of multi-part questions.",
    "instruction_following": "Explicit format and length constraints are often ignored; add constraint checks or few-shot examples.",
    "hallucination": "The model asserts unsupported content; penalize unsupported specifics and encourage hedging.",
    "style": "Cut conversational boilerplate (greetings, sign-offs) and match the requested layout.",
}

STRENGTH_TEXT = {
    "factual_accuracy": "Claims are reliably correct.",
    "relevance": "Stays on the question.",
    "completeness": "Covers the reference key points well.",
    "instruction_following": "Respects format and length constraints.",
    "hallucination": "Rarely introduces unsupported content.",
    "style": "Concise and cleanly formatted.",
}


def _examples(ds: Dataset, cons: dict, model: str, dim: str, worst: bool, k: int = 2) -> list[dict]:
    pmap = ds.prompt_by_id
    rows = [r for r in ds.responses if r.model == model and r.response_id in cons]
    rows.sort(key=lambda r: (cons[r.response_id][dim], r.response_id), reverse=not worst)
    out = []
    for r in rows[:k]:
        anns = ds.annotations_for(r.response_id)
        out.append(
            {
                "response_id": r.response_id,
                "prompt_id": r.prompt_id,
                "prompt": shorten(pmap[r.prompt_id].text, 120),
                "consensus_score": round(cons[r.response_id][dim], 2),
                "response_excerpt": shorten(r.text, 220),
                "annotator_rationales": {a.annotator_id: a.rationales.get(dim, "") for a in anns if a.source == "human"},
            }
        )
    return out


def model_feedback(
    ds: Dataset,
    comparison: dict,
    flagged: list[dict] | None = None,
    margin: float = 0.15,
    max_items: int = 3,
) -> dict[str, dict]:
    cons = consensus_scores(ds)
    dims_stats = comparison["dimensions"]
    models = comparison["models"]
    wm = win_matrix(comparison, "overall")
    flagged = flagged or []
    out = {}
    for m in models:
        dims = {}
        for d in DIMENSIONS:
            s = dims_stats[d][m]
            field = np.mean([dims_stats[d][o]["mean"] for o in models if o != m]) if len(models) > 1 else s["mean"]
            dims[d] = {**{k: round(v, 3) for k, v in s.items()}, "delta_vs_others": round(float(s["mean"] - field), 3)}

        by_delta = sorted(DIMENSIONS, key=lambda d: dims[d]["delta_vs_others"], reverse=True)
        strengths = [d for d in by_delta if dims[d]["delta_vs_others"] >= margin or dims[d]["mean"] >= 4.6][:max_items]
        weaknesses = [
            d for d in reversed(by_delta)
            if d not in strengths and (dims[d]["delta_vs_others"] <= -margin or dims[d]["mean"] < 4.0)
        ][:max_items]

        model_rids = [r.response_id for r in ds.responses if r.model == m and r.response_id in cons]
        halluc = [rid for rid in model_rids if cons[rid]["hallucination"] <= 2.5]
        if_viol = [rid for rid in model_rids if cons[rid]["instruction_following"] <= 3.0]
        out[m] = {
            "model": m,
            "rank": comparison["ranking"].index(m) + 1,
            "overall": {k: round(v, 3) for k, v in dims_stats["overall"][m].items()},
            "dimensions": dims,
            "strengths": [
                {"dimension": d, "mean": dims[d]["mean"], "delta_vs_others": dims[d]["delta_vs_others"],
                 "summary": STRENGTH_TEXT[d], "examples": _examples(ds, cons, m, d, worst=False, k=1)}
                for d in strengths
            ],
            "weaknesses": [
                {"dimension": d, "mean": dims[d]["mean"], "delta_vs_others": dims[d]["delta_vs_others"],
                 "advice": ADVICE[d], "examples": _examples(ds, cons, m, d, worst=True, k=2)}
                for d in weaknesses
            ],
            "win_rate_vs": {o: round(wm[m][o], 3) for o in models if o != m},
            "by_category": {c: round(v[m], 3) for c, v in comparison["by_category"].items()},
            "counts": {
                "responses": len(model_rids),
                "likely_hallucinations": len(halluc),
                "instruction_violations": len(if_viol),
                "flagged_for_adjudication": sum(1 for f in flagged if f["model"] == m),
            },
            "hallucination_examples": halluc[:5],
        }
    return out
