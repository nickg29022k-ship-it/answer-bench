"""End-to-end CLI run on the mock backend."""

import json

from typer.testing import CliRunner

from llm_response_benchmark.cli import app

from conftest import ROOT

runner = CliRunner()


def test_generate_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert runner.invoke(app, ["generate-data", "--out", str(a)]).exit_code == 0
    assert runner.invoke(app, ["generate-data", "--out", str(b)]).exit_code == 0
    for name in ("prompts", "responses", "annotations", "gold"):
        assert (a / f"{name}.jsonl").read_text() == (b / f"{name}.jsonl").read_text()
    # the checked-in data was produced by the same generator and seed
    assert (a / "annotations.jsonl").read_text() == (ROOT / "data" / "annotations.jsonl").read_text()


def test_full_pipeline(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    assert runner.invoke(app, ["generate-data", "--out", str(data)]).exit_code == 0
    (data / "rubric.yaml").write_text((ROOT / "data" / "rubric.yaml").read_text())

    res = runner.invoke(app, ["validate", "--data", str(data)])
    assert res.exit_code == 0, res.output
    assert "OK" in res.output

    for cmd in (["agreement"], ["calibrate"], ["compare", "--n-boot", "200"], ["disagreement", "--out", str(out / "dis")],
                ["annotate", "--backend", "mock", "--out", str(out / "auto.jsonl"), "--limit", "10"]):
        res = runner.invoke(app, [*cmd, "--data", str(data)])
        assert res.exit_code == 0, (cmd, res.output)
    assert len((out / "auto.jsonl").read_text().splitlines()) == 10
    assert (out / "dis" / "adjudication_queue.jsonl").exists()

    res = runner.invoke(app, ["run-all", "--data", str(data), "--out", str(out / "all"), "--n-boot", "300"])
    assert res.exit_code == 0, res.output
    all_ = out / "all"
    for f in ("report.md", "agreement.json", "calibration.json", "comparison.json", "flagged_items.json",
              "adjudication_queue.jsonl", "judge_agreement.json", "auto_annotations.jsonl"):
        assert (all_ / f).exists(), f
    fb = json.loads((all_ / "feedback" / "borealis-mini.json").read_text())
    assert fb["weaknesses"] and fb["rank"] == 3
    assert {w["dimension"] for w in fb["weaknesses"]} & {"factual_accuracy", "hallucination", "completeness"}
    report = (all_ / "report.md").read_text()
    for heading in ("Model comparison", "Inter-annotator agreement", "Annotator calibration",
                    "Disagreement workflow", "Per-model feedback"):
        assert heading in report
    queue = [json.loads(l) for l in (all_ / "adjudication_queue.jsonl").read_text().splitlines()]
    assert queue and queue[0]["queue_rank"] == 1
    assert all(q["priority"] >= queue[-1]["priority"] for q in queue)


def test_bad_backend_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = runner.invoke(app, ["annotate", "--data", str(ROOT / "data"), "--backend", "anthropic",
                              "--out", str(tmp_path / "x.jsonl")])
    assert res.exit_code == 2
    assert "ANTHROPIC_API_KEY" in res.output
