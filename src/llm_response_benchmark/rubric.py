"""Load and validate the YAML scoring rubric."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from .schemas import DIMENSIONS


class DimensionSpec(BaseModel):
    description: str
    weight: float = Field(default=1.0, ge=0)
    anchors: dict[int, str]

    @field_validator("anchors")
    @classmethod
    def _five_anchors(cls, v: dict[int, str]) -> dict[int, str]:
        if sorted(v) != [1, 2, 3, 4, 5]:
            raise ValueError(f"anchors must be defined for scores 1..5, got {sorted(v)}")
        return v


class Rubric(BaseModel):
    name: str
    version: int = 1
    scale: tuple[int, int] = (1, 5)
    dimensions: dict[str, DimensionSpec]

    @field_validator("dimensions")
    @classmethod
    def _all_dims(cls, v: dict[str, DimensionSpec]) -> dict[str, DimensionSpec]:
        if set(v) != set(DIMENSIONS):
            raise ValueError(
                f"rubric must define exactly {list(DIMENSIONS)}; missing {sorted(set(DIMENSIONS) - set(v))}, "
                f"extra {sorted(set(v) - set(DIMENSIONS))}"
            )
        return v

    @property
    def weights(self) -> dict[str, float]:
        return {d: s.weight for d, s in self.dimensions.items()}

    def as_prompt_text(self) -> str:
        """Render the rubric as plain text for an LLM judge prompt."""
        lines = []
        for d in DIMENSIONS:
            spec = self.dimensions[d]
            lines.append(f"{d}: {spec.description}")
            for score in range(5, 0, -1):
                lines.append(f"  {score} = {spec.anchors[score]}")
        return "\n".join(lines)


def load_rubric(path: str | Path) -> Rubric:
    with open(path, encoding="utf-8") as fh:
        return Rubric.model_validate(yaml.safe_load(fh))
