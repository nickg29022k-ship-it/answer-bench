"""Agreement metrics checked against published and hand-computed values."""

import math

import numpy as np
import pytest

from llm_response_benchmark.agreement import (
    cohen_kappa,
    coincidence_matrix,
    fleiss_kappa,
    fleiss_kappa_from_counts,
    krippendorff_alpha,
    pairwise_agreement,
)

nan = np.nan
# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 coders, 12 units, missing data.
KRIPP = np.array(
    [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
        [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
    ]
)


@pytest.mark.parametrize("metric,expected", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)])
def test_krippendorff_reference_example(metric, expected):
    assert krippendorff_alpha(KRIPP, metric) == pytest.approx(expected, abs=5e-4)


def test_krippendorff_hand_computed_nominal():
    # units: (1,1), (2,2), (1,2) -> o = [[2,1],[1,2]], n_c = [3,3], n = 6
    # D_o = 2/6, D_e = (3*3 + 3*3) / (6*5) = 0.6, alpha = 1 - (1/3)/0.6 = 4/9
    r = np.array([[1, 2, 1], [1, 2, 2]], dtype=float)
    values, o = coincidence_matrix(r)
    assert values.tolist() == [1.0, 2.0]
    assert o.tolist() == [[2.0, 1.0], [1.0, 2.0]]
    assert krippendorff_alpha(r, "nominal") == pytest.approx(4 / 9)
    # with only two categories, ordinal and nominal distances are proportional -> same alpha
    assert krippendorff_alpha(r, "ordinal") == pytest.approx(4 / 9)


def test_krippendorff_perfect_and_degenerate():
    r = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, nan]])
    assert krippendorff_alpha(r, "ordinal") == pytest.approx(1.0)
    assert math.isnan(krippendorff_alpha(np.full((2, 3), 3.0)))  # no variation at all
    # units with a single rating are not pairable and must be ignored
    r2 = np.array([[1, 2, 5], [1, 2, nan]])
    assert krippendorff_alpha(r2, "nominal") == pytest.approx(1.0)


def test_cohen_kappa_wikipedia_example():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> kappa = 0.4
    a = np.array([1] * 25 + [0] * 25)
    b = np.array([1] * 20 + [0] * 5 + [1] * 10 + [0] * 15)
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_weighted_hand_computed():
    a, b = np.array([1, 2, 3]), np.array([1, 3, 3])
    # unweighted: p_o = 2/3, p_e = 1/3 -> 0.5 ; quadratic: 1 - (0.25/3) / (1.25/3) = 0.8
    assert cohen_kappa(a, b, [1, 2, 3]) == pytest.approx(0.5)
    assert cohen_kappa(a, b, [1, 2, 3], weights="quadratic") == pytest.approx(0.8)
    # linear: weights |i-j|/2 -> sum w*O = 0.5/3, sum w*E = 1/3 * (1.5/3 + 2*1.5/3) = 0.5
    assert cohen_kappa(a, b, [1, 2, 3], weights="linear") == pytest.approx(1 - (0.5 / 3) / 0.5)


def test_cohen_kappa_drops_missing():
    a = np.array([1, 2, nan, 3])
    b = np.array([1, 2, 5, 3])
    assert cohen_kappa(a, b) == pytest.approx(1.0)


def test_fleiss_wikipedia_example():
    counts = np.array(
        [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
         [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    )
    assert fleiss_kappa_from_counts(counts) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_from_ratings_matches_counts_and_hand_value():
    # 3 raters, 2 units. unit1 all "1", unit2 = 1,2,2.
    r = np.array([[1, 1], [1, 2], [1, 2]], dtype=float)
    # P_1 = 1, P_2 = (1 + 4 - 3) / 6 = 1/3, P_bar = 2/3; p = [4/6, 2/6], P_e = 5/9
    expected = (2 / 3 - 5 / 9) / (1 - 5 / 9)
    assert fleiss_kappa(r, [1, 2]) == pytest.approx(expected)
    assert fleiss_kappa_from_counts(np.array([[3, 0], [1, 2]])) == pytest.approx(expected)
    with pytest.raises(ValueError):
        fleiss_kappa_from_counts(np.array([[3, 0], [1, 1]]))


def test_pairwise_agreement_values():
    r = np.array([[1, 2, 3, 4], [1, 3, 5, 4]], dtype=float)
    (p,) = pairwise_agreement(r, ["a", "b"], [1, 2, 3, 4, 5])
    assert p["n"] == 4
    assert p["exact"] == pytest.approx(0.5)
    assert p["within_1"] == pytest.approx(0.75)
    assert p["mean_abs_diff"] == pytest.approx(0.75)
    assert p["mean_signed_diff"] == pytest.approx(-0.75)


def test_agreement_report_on_synthetic(synth_ds):
    from llm_response_benchmark.agreement import agreement_report

    rep = agreement_report(synth_ds)
    dims = rep["dimensions"]
    assert set(dims) == {"factual_accuracy", "relevance", "completeness", "instruction_following", "hallucination", "style"}
    for v in dims.values():
        assert -1 <= v["krippendorff_alpha_ordinal"] <= 1
        assert len(v["pairwise"]) == 3
        # removing systematic annotator offsets should not make agreement worse
        assert v["krippendorff_alpha_interval_bias_corrected"] >= v["krippendorff_alpha_interval"] - 0.02
