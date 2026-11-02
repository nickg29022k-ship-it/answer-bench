import json

import pytest
from pydantic import ValidationError

from llm_response_benchmark.rubric import Rubric, load_rubric
from llm_response_benchmark.schemas import DIMENSIONS, Annotation, Constraints, Prompt, read_jsonl

from conftest import ROOT

FULL = {d: 3 for d in DIMENSIONS}


def test_annotation_valid():
    a = Annotation(annotator_id="x", response_id="r1", scores=FULL, rationales={"style": "ok"})
    assert a.source == "human"


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_annotation_score_out_of_range(bad):
    with pytest.raises(ValidationError):
        Annotation(annotator_id="x", response_id="r1", scores={**FULL, "style": bad})


def test_annotation_missing_dimension_and_extra_fields():
    scores = dict(FULL)
    scores.pop("hallucination")
    with pytest.raises(ValidationError, match="hallucination"):
        Annotation(annotator_id="x", response_id="r1", scores=scores)
    with pytest.raises(ValidationError):
        Annotation(annotator_id="x", response_id="r1", scores={**FULL, "humor": 3})
    with pytest.raises(ValidationError):
        Annotation(annotator_id="x", response_id="r1", scores=FULL, unexpected=True)


def test_constraints_validation():
    Constraints(format="bullets", num_bullets=3)
    with pytest.raises(ValidationError):
        Constraints(format="paragraph", num_bullets=3)
    with pytest.raises(ValidationError):
        Constraints(max_words=0)


def test_read_jsonl_reports_line_number(tmp_path):
    p = tmp_path / "prompts.jsonl"
    good = {"prompt_id": "p1", "category": "qa", "text": "hi"}
    p.write_text(json.dumps(good) + "\n\n" + json.dumps({"prompt_id": "p2"}) + "\n")
    with pytest.raises(ValueError, match=r"prompts.jsonl:3"):
        read_jsonl(p, Prompt)


def test_checked_in_data_validates():
    from llm_response_benchmark.dataset import Dataset

    ds = Dataset.load(ROOT / "data")
    assert ds.validate() == []
    assert len(ds.prompts) == 40
    assert len(ds.models) == 3
    assert len(ds.annotators) == 3
    assert len(ds.annotations) == 40 * 3 * 3


def test_rubric_loads_and_rejects_incomplete():
    rubric = load_rubric(ROOT / "data" / "rubric.yaml")
    assert set(rubric.dimensions) == set(DIMENSIONS)
    assert all(sorted(d.anchors) == [1, 2, 3, 4, 5] for d in rubric.dimensions.values())
    assert "5 =" in rubric.as_prompt_text()
    data = rubric.model_dump()
    data["dimensions"].pop("style")
    with pytest.raises(ValidationError):
        Rubric.model_validate(data)
    data = rubric.model_dump()
    data["dimensions"]["style"]["anchors"].pop(3)
    with pytest.raises(ValidationError):
        Rubric.model_validate(data)


def test_dataset_validate_catches_bad_references():
    from conftest import tiny_dataset

    ds = tiny_dataset({("p1-m1", "a"): {}, ("p1-m1", "b"): {}})
    assert ds.validate() == []
    ds.annotations.append(Annotation(annotator_id="a", response_id="nope", scores=FULL))
    ds.annotations.append(Annotation(annotator_id="a", response_id="p1-m1", scores=FULL))
    problems = ds.validate()
    assert any("unknown response nope" in p for p in problems)
    assert any("duplicate annotation" in p for p in problems)
