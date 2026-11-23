"""Acquisition strategies and annotation-budget enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from midogcx.data.patches import Region
from midogcx.selection.budget import AreaBudget, budget_from_fraction, select_under_budget
from midogcx.selection.strategies import (
    STRATEGIES,
    StrategyContext,
    rank_regions,
    rank_uncertainty,
    ranking_overlap,
)


def _region(rid: str, area_um2: float = 1e6, n_pos: int = 0) -> Region:
    return Region(
        region_id=rid,
        slide_id=1,
        x=0,
        y=0,
        width=512,
        height=512,
        tissue_px=1000,
        tissue_area_um2=area_um2,
        n_positives=n_pos,
    )


def test_budget_admits_until_exhausted():
    regions = [_region(f"r{i}") for i in range(5)]
    budget = AreaBudget(total_um2=3.0e6)
    selected = select_under_budget({r.region_id: r for r in regions}, [(f"r{i}", 0.0) for i in range(5)], budget)
    assert len(selected) == 3
    assert budget.used_um2 == pytest.approx(3.0e6)


def test_budget_skips_overshooting_region_and_continues():
    big = _region("big", area_um2=2.0e6)
    small = _region("small", area_um2=5e5)
    budget = AreaBudget(total_um2=2.5e6)
    budget.admit(big)
    ranking = [("small", 0.9), ("big", 0.8)]
    selected = select_under_budget({"big": big, "small": small}, ranking, budget)
    assert [r.region_id for r in selected] == ["small"]
    assert "big" in budget.skipped


def test_budget_is_never_exceeded_even_with_odd_sizes():
    regions = [_region(f"r{i}", area_um2=333_333.0) for i in range(10)]
    budget = budget_from_fraction(regions, 0.5)
    selected = select_under_budget(
        {r.region_id: r for r in regions}, [(r.region_id, 0.0) for r in regions], budget
    )
    assert budget.used_um2 <= budget.total_um2 + 1e-6
    assert len(selected) == 5


def test_duplicate_admission_is_refused():
    r = _region("r")
    budget = AreaBudget(total_um2=10e6)
    assert budget.admit(r) is True
    assert budget.admit(r) is False
    assert budget.used_um2 == pytest.approx(r.tissue_area_um2)


def test_budget_rejects_non_positive_total():
    with pytest.raises(ValueError):
        AreaBudget(total_um2=0)


def test_budget_fraction_validation():
    with pytest.raises(ValueError):
        budget_from_fraction([_region("r")], 0)
    with pytest.raises(ValueError):
        budget_from_fraction([_region("r")], 1.5)


def test_random_ranking_is_seeded_and_complete():
    regions = [_region(f"r{i}") for i in range(6)]
    ctx = StrategyContext(detections={}, threshold=0.5, seed=11)
    a = rank_regions(regions, "random", ctx)
    b = rank_regions(regions, "random", ctx)
    assert a == b
    assert {r for r, _ in a} == {r.region_id for r in regions}


def test_density_prefers_more_detections_per_area():
    sparse = _region("sparse", area_um2=1e6)
    dense = _region("dense", area_um2=1e6)
    detections = {
        "sparse": (np.zeros((1, 2)), np.array([0.9])),
        "dense": (np.zeros((5, 2)), np.array([0.9] * 5)),
    }
    ctx = StrategyContext(detections=detections, threshold=0.5)
    ranking = rank_regions([sparse, dense], "density", ctx)
    assert ranking[0][0] == "dense"


def test_uncertainty_prefers_borderline_detections():
    clear = _region("clear")
    unsure = _region("unsure")
    empty = _region("empty")
    detections = {
        "clear": (np.zeros((3, 2)), np.array([0.99, 0.98, 0.97])),
        "unsure": (np.zeros((3, 2)), np.array([0.52, 0.49, 0.48])),
        "empty": (np.zeros((0, 2)), np.zeros(0)),
    }
    ranking = rank_regions([clear, unsure, empty], "uncertainty", StrategyContext(detections, 0.5))
    assert ranking[0][0] == "unsure"
    # No detections must be handled explicitly, not crash or win.
    assert ranking[-1][0] in {"clear", "empty"}
    assert dict(ranking)["empty"] == 0.0


def test_uncertainty_with_no_detections_anywhere():
    regions = [_region("a"), _region("b")]
    ranking = rank_uncertainty(regions, {}, threshold=0.5)
    assert all(score == 0.0 for _, score in ranking)


def test_learned_strategies_use_the_predictor_output():
    class FakePredictor:
        feature_names = ("x",)

        def predict(self, X):
            return np.asarray([row[0] for row in X], dtype=float)

    regions = [_region("a"), _region("b"), _region("c")]
    ctx = StrategyContext(
        detections={},
        threshold=0.5,
        predictors={"count_error": FakePredictor(), "learned_loss": FakePredictor()},
        features={"a": {"x": 1.0}, "b": {"x": 5.0}, "c": {"x": 3.0}},
    )
    for name in ("count_error", "learned_loss"):
        ranking = rank_regions(regions, name, ctx)
        assert [r for r, _ in ranking] == ["b", "c", "a"]


def test_missing_features_raise_instead_of_silently_ranking():
    class FakePredictor:
        feature_names = ("x",)

        def predict(self, X):
            return np.zeros(len(X))

    ctx = StrategyContext(
        detections={}, threshold=0.5, predictors={"count_error": FakePredictor()}, features={"a": {"x": 1.0}}
    )
    with pytest.raises(KeyError):
        rank_regions([_region("a"), _region("b")], "count_error", ctx)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        rank_regions([_region("a")], "magic", StrategyContext({}, 0.5))


def test_strategy_names_are_the_five_plus_optional():
    assert STRATEGIES == ("random", "uncertainty", "density", "learned_loss", "count_error")


def test_ranking_overlap_metric():
    a = [("r1", 3.0), ("r2", 2.0), ("r3", 1.0)]
    b = [("r2", 3.0), ("r4", 2.0), ("r1", 1.0)]
    assert ranking_overlap(a, b, k=2) == pytest.approx(0.5)
