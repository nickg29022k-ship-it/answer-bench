"""Deterministic synthetic dataset generator.

Pipeline: seeds -> candidate responses assembled from reference facts (with planted omissions,
false claims, padding, off-topic tangents and format breaks, according to a per-model profile)
-> latent "true" rubric scores derived from those planted attributes -> noisy annotations
from three simulated annotators (one systematically lenient) with extra noise on ambiguous
prompts -> gold labels for a subset of responses.

The latent truth is written to `generation_truth.jsonl` so tests and the report can check
whether the disagreement workflow recovers what was planted. Nothing else in the pipeline reads it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schemas import (
    DIMENSIONS,
    Annotation,
    CandidateResponse,
    Constraints,
    GoldLabel,
    Prompt,
    ReferenceLabel,
    write_jsonl,
)
from .seeds import SEEDS
from .textutils import check_constraints, shorten

MODEL_PROFILES: dict[str, dict] = {
    # strong recall, chatty, tends to ignore format instructions
    "atlas-v2": dict(p_fact=0.90, p_false=0.12, p_filler=0.0, chatty=True, p_offtopic=0.20, p_format_break=0.35),
    # small model: misses facts and hallucinates more often
    "borealis-mini": dict(p_fact=0.55, p_false=0.35, p_filler=0.40, chatty=False, p_offtopic=0.25, p_format_break=0.25),
    # concise and obedient, occasionally incomplete
    "cirrus-large": dict(p_fact=0.75, p_false=0.05, p_filler=0.05, chatty=False, p_offtopic=0.03, p_format_break=0.06),
}

ANNOTATOR_PROFILES: dict[str, dict] = {
    "ann-01": dict(bias=0.0, leniency_slope=0.0, sd=0.3, slip=0.015),
    # lenient: inflates everything and overlooks problems (bias grows as true quality drops)
    "ann-02": dict(bias=0.7, leniency_slope=0.2, sd=0.3, slip=0.015),
    "ann-03": dict(bias=-0.1, leniency_slope=0.0, sd=0.45, slip=0.025),
}

SUBJECTIVE = {"relevance", "completeness", "instruction_following", "style"}

CHATTY_INTRO = {
    "factual_qa": "Great question!",
    "summarization": "Sure, here is a summary of the text.",
    "instruction_following": "Absolutely, here you go!",
    "explanation": "Great question, this is a classic topic.",
    "open_ended": "That's a great question, and lots of people wonder about it.",
}
CHATTY_OUTRO = [
    "I hope this helps! Let me know if you have any other questions.",
    "Feel free to ask if you would like more detail on any of this.",
]
FILLER = [
    "This is an important point to keep in mind.",
    "There are many more details that could be covered here as well.",
]
OFFTOPIC = {
    "factual_qa": "On a related note, the history of scientific discovery is full of surprising stories.",
    "summarization": "Similar announcements have become common in many places in recent years.",
    "instruction_following": "Of course, everyone's situation is different and preferences vary widely.",
    "explanation": "Many textbooks cover this topic in much greater depth across several chapters.",
    "open_ended": "People have debated questions like this one for a very long time.",
}


def _render(sentences: list[str], fmt: str, intro: str | None, outro: str | None) -> str:
    if fmt == "bullets":
        body = "\n".join(f"- {s}" for s in sentences)
    elif fmt == "lines":
        body = "\n".join(sentences)
    else:
        body = " ".join(sentences)
    parts = [p for p in (intro, body, outro) if p]
    sep = "\n" if fmt in ("bullets", "lines") else " "
    return sep.join(parts)


def _build_response(seed: dict, constraints: Constraints, profile: dict, rng: np.random.Generator) -> tuple[str, dict]:
    facts = seed["facts"]
    covered = [i for i in range(len(facts)) if rng.random() < profile["p_fact"]]
    if not covered:
        covered = [int(rng.integers(len(facts)))]
    has_false = bool(rng.random() < profile["p_false"])
    offtopic = bool(rng.random() < profile["p_offtopic"])
    filler = bool(rng.random() < profile["p_filler"])
    chatty = profile["chatty"]

    sentences = [facts[i] for i in covered]
    if has_false:
        sentences.insert(int(rng.integers(1, len(sentences) + 1)), seed["false"])
    if filler:
        sentences.append(FILLER[int(rng.integers(len(FILLER)))])
    if offtopic:
        sentences.append(OFFTOPIC[seed["category"]])

    natural = "lines" if "poem" in seed["prompt"].lower() else ("bullets" if constraints.format == "bullets" else "paragraph")
    fmt = natural
    format_break = False
    if constraints.format != "any" and rng.random() < profile["p_format_break"]:
        format_break = True
        fmt = "paragraph" if constraints.format == "bullets" else "bullets"

    intro = CHATTY_INTRO[seed["category"]] if chatty else None
    outro = CHATTY_OUTRO[int(rng.integers(len(CHATTY_OUTRO)))] if chatty else None
    text = _render(sentences, fmt, intro, outro)
    attrs = dict(
        covered=covered,
        n_facts=len(facts),
        has_false=has_false,
        offtopic=offtopic,
        filler=filler,
        chatty=chatty,
        format_break=format_break,
        violations=check_constraints(text, constraints),
    )
    return text, attrs


def _true_scores(attrs: dict, category: str, rng: np.random.Generator) -> dict[str, int]:
    n_cov = len(attrs["covered"])
    if attrs["has_false"]:
        fa = 2
    elif n_cov == 1:
        fa = 4
    else:
        fa = 5
    if attrs["has_false"]:
        hall = 1 if (category == "summarization" or rng.random() < 0.4) else 2
    else:
        hall = 5
    if n_cov == attrs["n_facts"]:
        comp = 5
    elif n_cov == attrs["n_facts"] - 1:
        comp = 4 if 0 in attrs["covered"] else 3
    else:
        comp = 2
    rel = 5 - 2 * attrs["offtopic"] - (1 if attrs["filler"] and attrs["chatty"] else 0)
    inst = max(1, 5 - 2 * len(attrs["violations"]))
    style = 5 - attrs["chatty"] - attrs["filler"] - attrs["offtopic"] - (1 if attrs["format_break"] else 0)
    return {
        "factual_accuracy": fa,
        "relevance": max(1, rel),
        "completeness": comp,
        "instruction_following": inst,
        "hallucination": hall,
        "style": max(2, style),
    }


def _rationale(dim: str, score: int, attrs: dict, seed: dict, ambiguous: bool, rng: np.random.Generator) -> str:
    missing = [seed["facts"][i] for i in range(attrs["n_facts"]) if i not in attrs["covered"]]
    if dim == "factual_accuracy":
        if attrs["has_false"]:
            if score <= 3:
                return f'Incorrect claim: "{shorten(seed["false"], 80)}"'
            return "Mostly accurate; one statement looked questionable but it is minor."
        return "Statements agree with the reference." if score >= 4 else "Nothing clearly wrong, but some wording is imprecise."
    if dim == "hallucination":
        if attrs["has_false"]:
            if score <= 3:
                return f'Asserts unsupported content: "{shorten(seed["false"], 80)}"'
            return "Possibly one unsupported detail, but nothing serious."
        return "No fabricated content noticed." if score >= 4 else "A detail seemed unsupported on first read."
    if dim == "completeness":
        if not missing:
            return "Covers all key points." if score >= 4 else "Covers the points but too briefly."
        if score >= 4:
            return "Covers the main points; a minor detail is missing."
        return "Missing: " + "; ".join(shorten(m, 60) for m in missing)
    if dim == "relevance":
        if attrs["offtopic"]:
            return "Includes an off-topic tangent." if score <= 3 else "Small tangent, otherwise on topic."
        if ambiguous and score <= 3:
            return "The prompt is open to interpretation and this reading feels only partly on target."
        return "Stays on topic." if score >= 4 else "Padding dilutes the answer."
    if dim == "instruction_following":
        if attrs["violations"]:
            if score <= 3:
                return "Violates instructions: " + "; ".join(attrs["violations"])
            return "Slightly off the requested format, acceptable overall."
        if ambiguous and score <= 3:
            return "Hard to say what the request expects; the answer only partially fits."
        return "Follows the requested format and length." if score >= 4 else "Technically compliant but misses the spirit of the request."
    if dim == "style":
        notes = []
        if attrs["chatty"]:
            notes.append("unnecessary pleasantries")
        if attrs["filler"]:
            notes.append("filler sentence")
        if attrs["format_break"]:
            notes.append("wrong layout")
        if notes and score <= 4:
            return "Style issues: " + ", ".join(notes) + "."
        if ambiguous:
            return ["Tone works for me.", "Tone is a matter of taste here.", "Reads a bit flat."][int(rng.integers(3))]
        return "Clear and concise." if score >= 4 else "Reads awkwardly."
    raise ValueError(dim)


def generate(out_dir: str | Path, seed: int = 7) -> dict[str, int]:
    """Generate the full synthetic benchmark into `out_dir`. Deterministic for a given seed."""
    out = Path(out_dir)
    rng = np.random.default_rng(seed)

    prompts: list[Prompt] = []
    references: list[ReferenceLabel] = []
    responses: list[CandidateResponse] = []
    annotations: list[Annotation] = []
    truth: list[dict] = []

    for k, s in enumerate(SEEDS, start=1):
        pid = f"p{k:03d}"
        constraints = Constraints(**s.get("constraints", {}))
        ambiguous = bool(s.get("ambiguous", False))
        prompts.append(
            Prompt(prompt_id=pid, category=s["category"], text=s["prompt"], constraints=constraints,
                   metadata={"source": "synthetic"})
        )
        references.append(
            ReferenceLabel(
                prompt_id=pid,
                reference_answer=" ".join(s["facts"]),
                key_facts=s["facts"],
                known_false_claims=[s["false"]],
                notes="Open-ended prompt; several readings are acceptable." if ambiguous else None,
            )
        )
        for model, profile in MODEL_PROFILES.items():
            rid = f"r-{pid}-{model.split('-')[0]}"
            text, attrs = _build_response(s, constraints, profile, rng)
            responses.append(CandidateResponse(response_id=rid, prompt_id=pid, model=model, text=text))
            true = _true_scores(attrs, s["category"], rng)

            # one deliberate "split decision" per ambiguous response
            split = None
            if ambiguous:
                split = (
                    list(ANNOTATOR_PROFILES)[int(rng.integers(len(ANNOTATOR_PROFILES)))],
                    sorted(SUBJECTIVE)[int(rng.integers(len(SUBJECTIVE)))],
                    int(rng.choice([-2, 2])),
                )
            for ann_id, ap in ANNOTATOR_PROFILES.items():
                scores, rationales = {}, {}
                for dim in DIMENSIONS:
                    t = true[dim]
                    x = t + ap["bias"] + ap["leniency_slope"] * (5 - t) + rng.normal(0, ap["sd"])
                    if ambiguous:
                        x += rng.normal(0, 1.3 if dim in SUBJECTIVE else 0.6)
                    if split and split[0] == ann_id and split[1] == dim:
                        x += split[2]
                    if rng.random() < ap["slip"]:
                        x += rng.choice([-2, 2])
                    sc = int(np.clip(np.rint(x), 1, 5))
                    scores[dim] = sc
                    rationales[dim] = _rationale(dim, sc, attrs, s, ambiguous, rng)
                annotations.append(
                    Annotation(annotator_id=ann_id, response_id=rid, scores=scores, rationales=rationales)
                )
            truth.append({"response_id": rid, "prompt_id": pid, "model": model, "ambiguous": ambiguous,
                          "true_scores": true, "attributes": attrs})

    # gold labels: expert scores (= latent truth) for a stratified sample of unambiguous responses
    clear = [t for t in truth if not t["ambiguous"]]
    pick = rng.choice(len(clear), size=min(24, len(clear)), replace=False)
    gold = [
        GoldLabel(response_id=clear[i]["response_id"], scores=clear[i]["true_scores"],
                  note="Adjudicated by expert reviewer.")
        for i in sorted(pick)
    ]

    counts = {
        "prompts": write_jsonl(out / "prompts.jsonl", prompts),
        "references": write_jsonl(out / "references.jsonl", references),
        "responses": write_jsonl(out / "responses.jsonl", responses),
        "annotations": write_jsonl(out / "annotations.jsonl", annotations),
        "gold": write_jsonl(out / "gold.jsonl", gold),
        "generation_truth": write_jsonl(out / "generation_truth.jsonl", truth),
    }
    return counts
