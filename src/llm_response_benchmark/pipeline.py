"""End-to-end pipeline: load -> agreement -> disagreement/calibration -> comparison -> feedback -> report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .agreement import agreement_report
from .auto_annotator import annotate, judge_vs_humans
from .compare import compare_models
from .dataset import Dataset
from .disagreement import DisagreementConfig, adjudication_queue, calibrate, flag_items
from .feedback import model_feedback
from .llm import get_backend
from .report import render_report
from .rubric import load_rubric
from .schemas import write_json, write_jsonl


def ambiguity_recovery(data_dir: Path, flagged: list[dict]) -> dict | None:
    """Compare flagged items against the generator's planted ambiguity (synthetic data only)."""
    path = Path(data_dir) / "generation_truth.jsonl"
    if not path.exists():
        return None
    truth = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            truth[rec["response_id"]] = rec["ambiguous"]
    planted = {r for r, a in truth.items() if a}
    amb = {f["response_id"] for f in flagged if f["label"] == "ambiguous"}
    anyf = {f["response_id"] for f in flagged}
    tp = len(amb & planted)
    return {
        "planted": len(planted),
        "flagged_ambiguous": len(amb),
        "true_positives": tp,
        "precision": tp / len(amb) if amb else float("nan"),
        "recall": tp / len(planted) if planted else float("nan"),
        "recall_any_flag": len(anyf & planted) / len(planted) if planted else float("nan"),
    }


def run_pipeline(
    data_dir: str | Path,
    out_dir: str | Path,
    rubric_path: str | Path | None = None,
    backend: str | None = "mock",
    n_boot: int = 2000,
    seed: int = 0,
    cfg: DisagreementConfig | None = None,
) -> dict:
    data_dir, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or DisagreementConfig()
    rubric = load_rubric(rubric_path or data_dir / "rubric.yaml")
    ds = Dataset.load(data_dir)
    problems = ds.validate()
    if problems:
        raise ValueError("dataset failed validation:\n" + "\n".join(problems))

    agree = agreement_report(ds)
    calib = calibrate(ds, cfg)
    flagged = flag_items(ds, cfg)
    queue = adjudication_queue(ds, cfg)
    comp = compare_models(ds, n_boot=n_boot, seed=seed, weights=rubric.weights)
    fb = model_feedback(ds, comp, flagged)
    recovery = ambiguity_recovery(data_dir, flagged)

    write_json(out / "agreement.json", agree)
    write_json(out / "calibration.json", calib)
    write_json(out / "flagged_items.json", {"config": asdict(cfg), "recovery_check": recovery, "items": flagged})
    write_jsonl(out / "adjudication_queue.jsonl", queue)
    write_json(out / "comparison.json", comp)
    for m, f in fb.items():
        write_json(out / "feedback" / f"{m}.json", f)

    judge = None
    if backend:
        be = get_backend(backend)
        auto, errors = annotate(ds, be, rubric)
        write_jsonl(out / "auto_annotations.jsonl", auto)
        judge = {"backend": be.name, "errors": errors, **judge_vs_humans(ds, auto)}
        write_json(out / "judge_agreement.json", judge)

    ctx = {
        "dataset": {
            "prompts": len(ds.prompts),
            "responses": len(ds.responses),
            "annotations": len(ds.annotations),
            "annotators": ds.annotators,
            "gold": len(ds.gold),
        },
        "comparison": comp,
        "agreement": agree,
        "calibration": calib,
        "flagged": flagged,
        "queue": queue,
        "feedback": fb,
        "config": asdict(cfg),
        "ambiguity_recovery": recovery,
        "judge": judge,
    }
    (out / "report.md").write_text(render_report(ctx), encoding="utf-8")
    return {
        "out_dir": str(out),
        "flagged": len(flagged),
        "queue": len(queue),
        "ranking": comp["ranking"],
        "judge_errors": len(judge["errors"]) if judge else 0,
    }
