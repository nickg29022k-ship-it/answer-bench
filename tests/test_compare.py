import numpy as np
import pytest

from llm_response_benchmark.compare import bootstrap_mean_ci, compare_models, win_matrix

from conftest import tiny_dataset


def test_bootstrap_ci_properties():
    m, lo, hi = bootstrap_mean_ci(np.full(20, 3.0))
    assert (m, lo, hi) == (3.0, 3.0, 3.0)
    x = np.random.default_rng(1).normal(4, 0.5, 200)
    m, lo, hi = bootstrap_mean_ci(x, n_boot=1000, seed=3)
    assert lo < m < hi
    assert hi - lo == pytest.approx(2 * 1.96 * x.std() / np.sqrt(len(x)), rel=0.25)
    assert bootstrap_mean_ci(x, seed=3) == bootstrap_mean_ci(x, seed=3)


def test_win_rates_hand_computed():
    # 4 prompts, single annotator pair. m1 beats m2 on p1,p2; ties on p3; loses on p4.
    s = {"p1": (5, 3), "p2": (4, 2), "p3": (3, 3), "p4": (2, 4)}
    table = {}
    for p, (a, b) in s.items():
        for ann in ("x", "y"):
            table[(f"{p}-m1", ann)] = {"factual_accuracy": a}
            table[(f"{p}-m2", ann)] = {"factual_accuracy": b}
    comp = compare_models(tiny_dataset(table), n_boot=200)
    w = next(x for x in comp["win_rates"] if x["dimension"] == "factual_accuracy")
    assert (w["model_a"], w["model_b"]) == ("m1", "m2")
    assert (w["wins"], w["losses"], w["ties"]) == (2, 1, 1)
    assert w["win_rate"] == pytest.approx((2 + 0.5) / 4)
    assert w["mean_diff"] == pytest.approx((2 + 2 + 0 - 2) / 4)
    assert comp["dimensions"]["factual_accuracy"]["m1"]["mean"] == pytest.approx(3.5)
    mat = win_matrix(comp, "factual_accuracy")
    assert mat["m2"]["m1"] == pytest.approx(1 - 0.625)
    # other dimensions default to 3 for everyone -> all ties
    style = next(x for x in comp["win_rates"] if x["dimension"] == "style")
    assert style["ties"] == 4 and not style["significant"]


def test_compare_on_synthetic_ranks_weak_model_last(synth_ds):
    comp = compare_models(synth_ds, n_boot=500)
    assert comp["ranking"][-1] == "borealis-mini"
    for d, per_model in comp["dimensions"].items():
        for s in per_model.values():
            assert s["ci_low"] <= s["mean"] <= s["ci_high"]
    # borealis was generated with the highest false-claim rate
    h = comp["dimensions"]["hallucination"]
    assert h["borealis-mini"]["mean"] < min(h["atlas-v2"]["mean"], h["cirrus-large"]["mean"])
