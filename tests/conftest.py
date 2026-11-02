from pathlib import Path

import pytest

from llm_response_benchmark.dataset import Dataset
from llm_response_benchmark.schemas import (
    DIMENSIONS,
    Annotation,
    CandidateResponse,
    Prompt,
    ReferenceLabel,
)
from llm_response_benchmark.synth import generate

ROOT = Path(__file__).resolve().parents[1]


def tiny_dataset(table: dict[tuple[str, str], dict[str, int]], models=("m1", "m2")) -> Dataset:
    """Build a Dataset from {(response_id, annotator): {dim: score}}; missing dims default to 3.

    Response ids look like '<prompt>-<model>'.
    """
    rids = sorted({r for r, _ in table})
    prompts = sorted({r.split("-")[0] for r in rids})
    return Dataset(
        prompts=[Prompt(prompt_id=p, category="qa", text=f"question {p}") for p in prompts],
        responses=[CandidateResponse(response_id=r, prompt_id=r.split("-")[0], model=r.split("-")[1], text=f"answer {r}")
                   for r in rids],
        references=[ReferenceLabel(prompt_id=p, reference_answer="ref", key_facts=["fact"]) for p in prompts],
        annotations=[
            Annotation(annotator_id=a, response_id=r, scores={d: s.get(d, 3) for d in DIMENSIONS})
            for (r, a), s in table.items()
        ],
    )


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("data")
    generate(d, seed=7)
    (d / "rubric.yaml").write_text((ROOT / "data" / "rubric.yaml").read_text())
    return d


@pytest.fixture(scope="session")
def synth_ds(synth_dir) -> Dataset:
    return Dataset.load(synth_dir)
