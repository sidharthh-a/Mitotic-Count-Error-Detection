"""Detection/counting metric tests, including the FP/FN cancellation case."""

from __future__ import annotations

import pytest

from midogcx.evaluation.metrics import (
    SpecimenMetrics,
    aggregate_metrics,
    macro_average,
    paired_comparison,
    pooled_counts,
)


def _m(sid, tp, fp, fn):
    return SpecimenMetrics(
        slide_id=sid,
        n_pred=tp + fp,
        n_gt=tp + fn,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def test_precision_recall_f1():
    m = _m(1, tp=8, fp=2, fn=2)
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.8)
    assert m.f1 == pytest.approx(0.8)


def test_count_error_is_absolute_and_bias_is_signed():
    over = _m(1, tp=10, fp=5, fn=0)
    under = _m(2, tp=10, fp=0, fn=5)
    assert over.abs_count_error == 5 and over.signed_bias == 5
    assert under.abs_count_error == 5 and under.signed_bias == -5


def test_cancellation_case_matches_documented_example():
    # Specimen 404 in the blueprint: 64 predictions, 61 ground truth, TP 19, FP 45, FN 42.
    m = SpecimenMetrics(
        slide_id=404, n_pred=64, n_gt=61, true_positives=19, false_positives=45, false_negatives=42
    )
    assert m.abs_count_error == 3
    assert m.f1 < 0.31
    # Counting error is small, localisation is not: exactly why both must be reported.
    assert m.false_positives > 40 and m.false_negatives > 40


def test_empty_specimen_uses_epsilon_convention():
    m = _m(9, tp=0, fp=0, fn=0)
    assert m.precision == pytest.approx(0.0)
    assert m.f1 == pytest.approx(0.0)


def test_pooled_counts_sum_then_divide():
    results = [_m(1, 8, 2, 2), _m(2, 6, 4, 4)]
    pooled = pooled_counts(results)
    assert pooled["tp"] == 14 and pooled["fp"] == 6 and pooled["fn"] == 6
    assert pooled["precision"] == pytest.approx(14 / 20)
    assert pooled["recall"] == pytest.approx(14 / 20)


def test_macro_average_differs_from_pooled_on_imbalanced_specimens():
    results = [_m(1, 40, 0, 0), _m(2, 0, 10, 10)]
    macro = macro_average(results)
    pooled = pooled_counts(results)
    # Pooled recall is dominated by the easy specimen; macro weights both equally.
    assert pooled["recall"] > 0.7
    assert macro["recall"] < 0.6


def test_mae_and_bias_aggregation():
    results = [_m(1, 10, 0, 0), _m(2, 8, 0, 2)]
    agg = aggregate_metrics(results, n_boot=50, seed=0)
    assert agg["counting"]["mae"] == pytest.approx(1.0)
    assert agg["counting"]["mean_signed_bias"] == pytest.approx(-1.0)


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    results = [_m(i, 10 - i, i, i) for i in range(1, 12)]
    a = aggregate_metrics(results, n_boot=200, seed=42)
    b = aggregate_metrics(results, n_boot=200, seed=42)
    assert a["counting_ci95"] == b["counting_ci95"]


def test_paired_comparison_requires_identical_specimens():
    with pytest.raises(ValueError):
        paired_comparison([_m(1, 5, 1, 1)], [_m(2, 5, 1, 1)], n_boot=10)


def test_paired_comparison_reports_improvement_direction():
    # better: 10 predictions vs 9 GT -> abs error 1; worse: 10 vs 5 -> abs error 5.
    better = [_m(1, 9, 1, 0), _m(2, 9, 1, 0)]
    worse = [_m(1, 5, 5, 0), _m(2, 5, 5, 0)]
    out = paired_comparison(better, worse, n_boot=100, seed=0)
    # Fewer abs-count-error for `better` -> negative delta.
    assert out["delta_abs_count_error"]["mean"] < 0
    assert out["delta_f1"]["mean"] > 0
