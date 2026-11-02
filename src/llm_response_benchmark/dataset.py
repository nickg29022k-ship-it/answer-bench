"""Loading a benchmark directory and turning annotations into score matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schemas import (
    DIMENSIONS,
    Annotation,
    CandidateResponse,
    GoldLabel,
    Prompt,
    ReferenceLabel,
    read_jsonl,
)

FILES = {
    "prompts": "prompts.jsonl",
    "responses": "responses.jsonl",
    "references": "references.jsonl",
    "annotations": "annotations.jsonl",
    "gold": "gold.jsonl",
}


@dataclass
class Dataset:
    prompts: list[Prompt]
    responses: list[CandidateResponse]
    references: list[ReferenceLabel]
    annotations: list[Annotation]
    gold: list[GoldLabel] = field(default_factory=list)

    # ---- loading -------------------------------------------------------
    @classmethod
    def load(cls, data_dir: str | Path, annotations_file: str | Path | None = None) -> Dataset:
        d = Path(data_dir)
        gold_path = d / FILES["gold"]
        ann_path = Path(annotations_file) if annotations_file else d / FILES["annotations"]
        return cls(
            prompts=read_jsonl(d / FILES["prompts"], Prompt),
            responses=read_jsonl(d / FILES["responses"], CandidateResponse),
            references=read_jsonl(d / FILES["references"], ReferenceLabel),
            annotations=read_jsonl(ann_path, Annotation),
            gold=read_jsonl(gold_path, GoldLabel) if gold_path.exists() else [],
        )

    # ---- lookups -------------------------------------------------------
    @property
    def prompt_by_id(self) -> dict[str, Prompt]:
        return {p.prompt_id: p for p in self.prompts}

    @property
    def response_by_id(self) -> dict[str, CandidateResponse]:
        return {r.response_id: r for r in self.responses}

    @property
    def reference_by_prompt(self) -> dict[str, ReferenceLabel]:
        return {r.prompt_id: r for r in self.references}

    @property
    def models(self) -> list[str]:
        return sorted({r.model for r in self.responses})

    @property
    def annotators(self) -> list[str]:
        return sorted({a.annotator_id for a in self.annotations})

    @property
    def annotated_response_ids(self) -> list[str]:
        seen = {a.response_id for a in self.annotations}
        return [r.response_id for r in self.responses if r.response_id in seen]

    # ---- validation ----------------------------------------------------
    def validate(self) -> list[str]:
        """Cross-file integrity checks. Returns a list of human-readable problems."""
        problems: list[str] = []

        def dupes(ids: list[str], what: str) -> None:
            seen: set[str] = set()
            for i in ids:
                if i in seen:
                    problems.append(f"duplicate {what}: {i}")
                seen.add(i)

        dupes([p.prompt_id for p in self.prompts], "prompt_id")
        dupes([r.response_id for r in self.responses], "response_id")
        dupes([r.prompt_id for r in self.references], "reference prompt_id")
        dupes([f"{a.annotator_id}/{a.response_id}" for a in self.annotations], "annotation")

        pids = set(self.prompt_by_id)
        rids = set(self.response_by_id)
        for r in self.responses:
            if r.prompt_id not in pids:
                problems.append(f"response {r.response_id} references unknown prompt {r.prompt_id}")
        for ref in self.references:
            if ref.prompt_id not in pids:
                problems.append(f"reference for unknown prompt {ref.prompt_id}")
        for p in pids - {r.prompt_id for r in self.references}:
            problems.append(f"prompt {p} has no reference label")
        for a in self.annotations:
            if a.response_id not in rids:
                problems.append(f"annotation by {a.annotator_id} for unknown response {a.response_id}")
        for g in self.gold:
            if g.response_id not in rids:
                problems.append(f"gold label for unknown response {g.response_id}")
        counts: dict[str, int] = {}
        for a in self.annotations:
            counts[a.response_id] = counts.get(a.response_id, 0) + 1
        single = [rid for rid, c in counts.items() if c < 2]
        if single:
            problems.append(
                f"{len(single)} responses have fewer than 2 annotations (agreement undefined for them)"
            )
        return problems

    # ---- matrices ------------------------------------------------------
    def score_matrix(
        self,
        dimension: str,
        annotators: list[str] | None = None,
        response_ids: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Return (matrix[annotator, response], annotator_ids, response_ids). Missing = NaN."""
        if dimension not in DIMENSIONS:
            raise ValueError(f"unknown dimension {dimension}")
        annotators = annotators or self.annotators
        response_ids = response_ids or self.annotated_response_ids
        a_idx = {a: i for i, a in enumerate(annotators)}
        r_idx = {r: j for j, r in enumerate(response_ids)}
        m = np.full((len(annotators), len(response_ids)), np.nan)
        for ann in self.annotations:
            i, j = a_idx.get(ann.annotator_id), r_idx.get(ann.response_id)
            if i is None or j is None:
                continue
            m[i, j] = ann.scores[dimension]
        return m, annotators, response_ids

    def annotations_for(self, response_id: str) -> list[Annotation]:
        return [a for a in self.annotations if a.response_id == response_id]
