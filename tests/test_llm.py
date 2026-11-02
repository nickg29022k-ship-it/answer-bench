import json

import pytest

from llm_response_benchmark.auto_annotator import annotate, build_judge_prompt, judge_vs_humans, parse_judge_output
from llm_response_benchmark.llm import BackendError, MockBackend, get_backend
from llm_response_benchmark.rubric import load_rubric
from llm_response_benchmark.schemas import DIMENSIONS
from llm_response_benchmark.textutils import check_constraints, word_count
from llm_response_benchmark.schemas import Constraints

from conftest import ROOT


def test_get_backend_default_and_unknown():
    assert get_backend("mock").name == "mock"
    with pytest.raises(BackendError, match="unknown backend"):
        get_backend("gpt-17")


def test_network_backends_fail_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendError, match="ANTHROPIC_API_KEY"):
        get_backend("anthropic")
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        get_backend("openai")
    # nothing listens on port 9 locally; the error should explain how to fix it
    be = get_backend("ollama", host="http://127.0.0.1:9")
    with pytest.raises(BackendError, match="ollama serve"):
        be.complete("hi")


def test_mock_is_deterministic(synth_ds):
    rubric = load_rubric(ROOT / "data" / "rubric.yaml")
    ds = synth_ds
    r = ds.responses[0]
    prompt = build_judge_prompt(ds.prompt_by_id[r.prompt_id], ds.reference_by_prompt[r.prompt_id], r, rubric)
    a = MockBackend().complete(prompt)
    assert a == MockBackend().complete(prompt)
    scores, rationales = parse_judge_output(a)
    assert set(scores) == set(DIMENSIONS)
    assert MockBackend().complete("hello there").startswith("[mock completion]")


def test_mock_judge_detects_planted_problems(synth_dir, synth_ds):
    truth = {json.loads(l)["response_id"]: json.loads(l) for l in open(synth_dir / "generation_truth.jsonl")}
    anns, errors = annotate(synth_ds, get_backend("mock"), load_rubric(ROOT / "data" / "rubric.yaml"))
    assert errors == [] and len(anns) == len(synth_ds.responses)
    for a in anns:
        t = truth[a.response_id]["attributes"]
        if t["has_false"]:
            assert a.scores["hallucination"] <= 3
    rep = judge_vs_humans(synth_ds, anns)
    assert rep["n"] == len(anns)
    assert rep["dimensions"]["hallucination"]["alpha_ordinal_vs_median"] > 0.6


def test_parse_judge_output_tolerates_wrapping():
    body = {"scores": {d: 4 for d in DIMENSIONS}, "rationales": {"style": "fine"}}
    text = "Here you go:\n```json\n" + json.dumps(body) + "\n```"
    scores, rats = parse_judge_output(text)
    assert scores["style"] == 4 and rats == {"style": "fine"}
    body["scores"]["style"] = 9
    assert parse_judge_output(json.dumps(body))[0]["style"] == 5  # clipped into range
    with pytest.raises(ValueError):
        parse_judge_output("no json here")
    del body["scores"]["relevance"]
    with pytest.raises(ValueError, match="relevance"):
        parse_judge_output(json.dumps(body))


def test_constraint_checks():
    bullets = "- one\n- two\n- three\n- four"
    assert check_constraints(bullets, Constraints(format="bullets", num_bullets=3)) == ["expected 3 bullets, got 4"]
    assert check_constraints("A paragraph.", Constraints(format="bullets")) == ["expected bullet points"]
    assert check_constraints(bullets, Constraints(format="paragraph")) == ["expected a paragraph, got bullets"]
    assert word_count("- one two\n- three") == 3
    assert check_constraints("a b c d", Constraints(max_words=3)) == ["over word limit (4 > 3)"]
