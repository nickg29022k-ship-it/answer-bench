"""Automated annotator: an LLM judge that fills in the same Annotation schema as humans."""

from __future__ import annotations

import json
import re

import numpy as np

from .agreement import cohen_kappa, krippendorff_alpha
from .dataset import Dataset
from .llm import LLMBackend
from .rubric import Rubric
from .schemas import DIMENSIONS, Annotation, CandidateResponse, Prompt, ReferenceLabel

JUDGE_SYSTEM = (
    "You are a meticulous evaluator of AI-generated answers. Score the candidate response on each "
    "rubric dimension using only integers 1-5 and the anchor descriptions. Reply with a single JSON "
    'object: {"scores": {<dimension>: int}, "rationales": {<dimension>: "one sentence"}}. No other text.'
)


def build_judge_prompt(prompt: Prompt, ref: ReferenceLabel, response: CandidateResponse, rubric: Rubric) -> str:
    facts = "\n".join(f"- {f}" for f in ref.key_facts)
    falses = "\n".join(f"- {f}" for f in ref.known_false_claims)
    return (
        "Evaluate the candidate response below.\n\n"
        f"<rubric>\n{rubric.as_prompt_text()}\n</rubric>\n\n"
        f"<prompt>\n{prompt.text}\n</prompt>\n\n"
        f"<constraints>\n{prompt.constraints.model_dump_json()}\n</constraints>\n\n"
        f"<reference_answer>\n{ref.reference_answer}\n</reference_answer>\n\n"
        f"<key_facts>\n{facts}\n</key_facts>\n\n"
        f"<false_claims>\n{falses}\n</false_claims>\n\n"
        f"<response>\n{response.text}\n</response>\n\n"
        f"Dimensions to score: {', '.join(DIMENSIONS)}. Return JSON only."
    )


def parse_judge_output(text: str) -> tuple[dict[str, int], dict[str, str]]:
    """Extract scores and rationales from a judge reply. Tolerates prose or code fences around the JSON."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    data = json.loads(m.group(0))
    raw_scores = data.get("scores", data)
    scores = {}
    for d in DIMENSIONS:
        if d not in raw_scores:
            raise ValueError(f"judge output missing dimension {d}")
        v = int(round(float(raw_scores[d])))
        scores[d] = min(5, max(1, v))
    rationales = {d: str(r) for d, r in (data.get("rationales") or {}).items() if d in DIMENSIONS}
    return scores, rationales


def annotate(
    ds: Dataset, backend: LLMBackend, rubric: Rubric, limit: int | None = None, annotator_id: str | None = None
) -> tuple[list[Annotation], list[str]]:
    """Run the judge over all responses. Returns (annotations, errors)."""
    pmap, rmap = ds.prompt_by_id, ds.reference_by_prompt
    ann_id = annotator_id or f"auto-{backend.name}"
    out, errors = [], []
    for resp in ds.responses[:limit] if limit else ds.responses:
        text = build_judge_prompt(pmap[resp.prompt_id], rmap[resp.prompt_id], resp, rubric)
        try:
            reply = backend.complete(text, system=JUDGE_SYSTEM, temperature=0.0)
            scores, rationales = parse_judge_output(reply)
        except ValueError as exc:
            errors.append(f"{resp.response_id}: {exc}")
            continue
        out.append(Annotation(annotator_id=ann_id, response_id=resp.response_id, scores=scores,
                              rationales=rationales, source="auto"))
    return out, errors


def judge_vs_humans(ds: Dataset, auto: list[Annotation]) -> dict:
    """Agreement between the automated annotator and the (rounded) human consensus, per dimension."""
    human = [a for a in ds.annotations if a.source == "human"]
    by_resp: dict[str, list[Annotation]] = {}
    for a in human:
        by_resp.setdefault(a.response_id, []).append(a)
    auto_map = {a.response_id: a for a in auto}
    rids = [r for r in auto_map if r in by_resp]
    out = {"n": len(rids), "dimensions": {}}
    for d in DIMENSIONS:
        h = np.array([np.median([a.scores[d] for a in by_resp[r]]) for r in rids], dtype=float)
        j = np.array([auto_map[r].scores[d] for r in rids], dtype=float)
        hr = np.rint(h)
        out["dimensions"][d] = {
            "alpha_ordinal_vs_median": krippendorff_alpha(np.vstack([hr, j]), "ordinal"),
            "kappa_quadratic": cohen_kappa(hr, j, [1, 2, 3, 4, 5], weights="quadratic"),
            "mae": float(np.mean(np.abs(h - j))) if len(rids) else float("nan"),
            "exact": float(np.mean(hr == j)) if len(rids) else float("nan"),
            "within_1": float(np.mean(np.abs(hr - j) <= 1)) if len(rids) else float("nan"),
            "mean_bias": float(np.mean(j - h)) if len(rids) else float("nan"),
        }
    return out
