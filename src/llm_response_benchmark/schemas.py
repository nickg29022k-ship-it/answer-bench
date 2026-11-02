"""Pydantic models for every JSONL record type in the benchmark, plus I/O helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DIMENSIONS: tuple[str, ...] = (
    "factual_accuracy",
    "relevance",
    "completeness",
    "instruction_following",
    "hallucination",
    "style",
)

Dimension = Literal[
    "factual_accuracy",
    "relevance",
    "completeness",
    "instruction_following",
    "hallucination",
    "style",
]

Score = Annotated[int, Field(ge=1, le=5)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Constraints(_Strict):
    """Machine-checkable instructions attached to a prompt."""

    max_words: int | None = Field(default=None, gt=0)
    format: Literal["any", "paragraph", "bullets"] = "any"
    num_bullets: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _bullets_need_bullet_format(self) -> Constraints:
        if self.num_bullets is not None and self.format != "bullets":
            raise ValueError("num_bullets requires format='bullets'")
        return self


class Prompt(_Strict):
    prompt_id: str
    category: str
    text: str = Field(min_length=1)
    constraints: Constraints = Field(default_factory=Constraints)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateResponse(_Strict):
    response_id: str
    prompt_id: str
    model: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReferenceLabel(_Strict):
    """Reference information for a prompt: what a good answer must and must not contain."""

    prompt_id: str
    reference_answer: str
    key_facts: list[str] = Field(min_length=1)
    known_false_claims: list[str] = Field(default_factory=list)
    notes: str | None = None


class Annotation(_Strict):
    """One annotator's judgement of one candidate response on all rubric dimensions."""

    annotator_id: str
    response_id: str
    scores: dict[Dimension, Score]
    rationales: dict[Dimension, str] = Field(default_factory=dict)
    overall_comment: str | None = None
    source: Literal["human", "auto"] = "human"

    @field_validator("scores")
    @classmethod
    def _all_dimensions(cls, v: dict[str, int]) -> dict[str, int]:
        missing = set(DIMENSIONS) - set(v)
        if missing:
            raise ValueError(f"missing scores for dimensions: {sorted(missing)}")
        return v


class GoldLabel(_Strict):
    """Expert-adjudicated scores for a subset of responses, used to calibrate annotators."""

    response_id: str
    scores: dict[Dimension, Score]
    adjudicator: str = "expert"
    note: str | None = None

    @field_validator("scores")
    @classmethod
    def _all_dimensions(cls, v: dict[str, int]) -> dict[str, int]:
        missing = set(DIMENSIONS) - set(v)
        if missing:
            raise ValueError(f"missing scores for dimensions: {sorted(missing)}")
        return v


T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path, model: type[T]) -> list[T]:
    """Parse a JSONL file into a list of `model` instances. Errors include the line number."""
    out: list[T] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(model.model_validate_json(line))
            except Exception as exc:  # re-raise with location
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return out


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            if isinstance(rec, BaseModel):
                fh.write(rec.model_dump_json(exclude_none=False))
            else:
                fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, BaseModel):
        return o.model_dump()
    raise TypeError(f"not JSON serializable: {type(o)}")
