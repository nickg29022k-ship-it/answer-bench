import numpy as np
import pytest

from llm_response_benchmark.disagreement import (
    DisagreementConfig,
    adjudication_queue,
    calibrate,
    consensus_scores,
    fit_additive_biases,
    flag_items,
    item_disagreement,
    loo_consensus,
)
from llm_response_benchmark.pipeline import ambiguity_recovery

from conftest import tiny_dataset


def test_loo_consensus_hand_values():
    m = np.array([[1.0, 4.0], [3.0, np.nan], [5.0, 2.0]])
    loo = loo_consensus(m)
    assert loo[:, 0].tolist() == [4.0, 3.0, 2.0]
    assert loo[0, 1] == 2.0 and loo[2, 1] == 4.0


def test_additive_bias_recovers_offsets():
    rng = np.random.default_rng(0)
    quality = rng.uniform(1.5, 3.5, size=60)
    offsets = np.array([0.0, 1.0, -0.3])
    m = quality[None, :] + offsets[:, None]
    b = fit_additive_biases(m)
    # median-centred: the middle annotator (offset 0.0) is the reference
    assert b == pytest.approx(offsets, abs=1e-6)
    m[1, 5] = np.nan
    assert fit_additive_biases(m) == pytest.approx(offsets, abs=1e-6)


def _three_annotator_table():
    table = {}
    for p in range(1, 9):
        for model in ("m1", "m2"):
            rid = f"p{p}-{model}"
            base = 2 + (p % 3)
            table[(rid, "a")] = {d: base for d in ("factual_accuracy", "style")}
            table[(rid, "b")] = {d: base for d in ("factual_accuracy", "style")}
            table[(rid, "c")] = {d: min(5, base + 1) for d in ("factual_accuracy", "style")}  # lenient
    # one genuinely contested response: annotators split on two dimensions
    table[("p1-m1", "a")] = {"factual_accuracy": 1, "style": 5, "relevance": 1}
    table[("p1-m1", "b")] = {"factual_accuracy": 4, "style": 2, "relevance": 5}
    table[("p1-m1", "c")] = {"factual_accuracy": 5, "style": 3, "relevance": 3}
    return table


def test_flagging_and_queue_on_constructed_data():
    ds = tiny_dataset(_three_annotator_table())
    recs = item_disagreement(ds)
    fa = next(r for r in recs if r["response_id"] == "p1-m1" and r["dimension"] == "factual_accuracy")
    assert fa["raw_range"] == 4 and fa["disputed"]
    flagged = flag_items(ds)
    assert flagged[0]["response_id"] == "p1-m1"
    assert flagged[0]["label"] == "ambiguous"
    # the lenient annotator's +1 everywhere must not cause flags once bias is removed
    assert {f["response_id"] for f in flagged} == {"p1-m1"}
    # ... but without bias correction the systematic offset alone is not enough to flag either
    raw = flag_items(ds, DisagreementConfig(bias_correct=False))
    assert raw[0]["response_id"] == "p1-m1"

    queue = adjudication_queue(ds)
    task = queue[0]
    assert task["queue_rank"] == 1 and task["status"] == "pending" and task["resolution"] is None
    dims = {d["dimension"]: d for d in task["disputed"]}
    assert dims["factual_accuracy"]["scores"] == {"a": 1, "b": 4, "c": 5}
    assert set(dims["factual_accuracy"]["rationales"]) == {"a", "b", "c"}


def test_outlier_annotator_label():
    table = {}
    for p in range(1, 6):
        for a in "abc":
            table[(f"p{p}-m1", a)] = {"style": 4}
    table[("p2-m1", "c")] = {"style": 1}
    ds = tiny_dataset(table)
    flagged = flag_items(ds, DisagreementConfig(bias_correct=False))
    assert len(flagged) == 1
    assert flagged[0]["label"] == "ambiguous"  # a 3-point split on one dimension is severe
    assert flagged[0]["outlier_annotators"] == ["c"]
    table[("p2-m1", "c")] = {"style": 2}
    flagged = flag_items(tiny_dataset(table), DisagreementConfig(bias_correct=False))
    assert flagged[0]["label"] == "outlier_annotator"


def test_consensus_scores_bias_correction():
    ds = tiny_dataset(_three_annotator_table())
    raw = consensus_scores(ds)
    corr = consensus_scores(ds, bias_correct=True)
    # p2: a=b=4, c=5 -> raw mean 4.333; corrected removes c's +1 offset -> 4.0
    assert raw["p2-m1"]["factual_accuracy"] == pytest.approx(13 / 3)
    assert corr["p2-m1"]["factual_accuracy"] == pytest.approx(4.0, abs=0.15)


def test_calibration_flags_lenient_annotator(synth_ds):
    rep = calibrate(synth_ds)
    ann = rep["annotators"]
    assert "lenient" in ann["ann-02"]["flags"]
    assert "lenient" not in ann["ann-01"]["flags"] and "harsh" not in ann["ann-01"]["flags"]
    assert ann["ann-02"]["overall"]["bias"] > ann["ann-01"]["overall"]["bias"] + 0.3
    assert ann["ann-02"]["overall"]["gold_bias"] > 0.2
    assert ann["ann-01"]["overall"]["gold_mae"] < ann["ann-02"]["overall"]["gold_mae"]


def test_ambiguity_recovery_on_synthetic(synth_dir, synth_ds):
    rec = ambiguity_recovery(synth_dir, flag_items(synth_ds))
    assert rec["planted"] == 18
    assert rec["precision"] >= 0.6
    assert rec["recall"] >= 0.6
    assert rec["recall_any_flag"] >= 0.8
