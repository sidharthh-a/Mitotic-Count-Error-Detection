"""Regression tests for the acquisition predictors.

Bug under test: ``n_splits`` was read from the ``acquisition.error_predictor`` config block
for grouping folds, but the same block was also passed wholesale as estimator
hyperparameters, so ``HistGradientBoostingRegressor.__init__()`` received ``n_splits`` and
raised ``TypeError`` before the baseline gate could evaluate anything.

The fix separates cross-validation settings from estimator hyperparameters at both the
``fit_predictors`` boundary and inside ``RegionRegressor``. These tests pin that separation,
the grouped-fold behaviour it must not break, and the shipped config that triggered it.
"""

from __future__ import annotations

import numpy as np
import pytest

from midogcx.acquisition import fit_predictors
from midogcx.config import load_config
from midogcx.models.error_predictor import (
    CV_ONLY_PARAMS,
    FEATURE_NAMES,
    RegionRegressor,
    feature_matrix,
    split_cv_params,
)


def _synthetic_regions(n_specimens: int = 4, per_specimen: int = 10, seed: int = 0):
    """Feature rows with specimen groups, plus count-error and loss targets."""
    rng = np.random.default_rng(seed)
    n = n_specimens * per_specimen
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    groups = np.repeat([f"specimen_{i}" for i in range(n_specimens)], per_specimen)
    # Targets that actually depend on the features, so out-of-fold scoring is meaningful.
    y_count = np.abs(X[:, 0]) + 0.1 * rng.normal(size=n)
    y_loss = np.abs(X[:, 1]) + 0.1 * rng.normal(size=n)
    return X, y_count, y_loss, groups


def _config_block() -> dict:
    """The predictor block exactly as shipped in ``configs/base.yaml``."""
    return dict(load_config()["acquisition"]["error_predictor"])


def test_shipped_config_block_contains_cv_key():
    """Guards the premise of the bug: the config really does mix CV and estimator keys."""
    block = _config_block()
    assert "n_splits" in block
    assert set(block) & CV_ONLY_PARAMS


def test_split_cv_params_separates_fold_count():
    params, n_splits = split_cv_params({"max_depth": 2, "n_splits": 3})
    assert params == {"max_depth": 2}
    assert n_splits == 3

    # An explicit argument overrides the config value.
    params, n_splits = split_cv_params({"n_splits": 3}, n_splits=7)
    assert params == {}
    assert n_splits == 7

    # Missing config falls back to the documented default.
    assert split_cv_params(None) == ({}, 5)


def test_regressor_accepts_shipped_config_block():
    """The original failure: constructing and fitting with the shipped block must work."""
    X, y_count, _, groups = _synthetic_regions()
    model = RegionRegressor(params=_config_block(), target_name="abs_count_error")

    report = model.fit(X, y_count, groups=groups)

    assert "n_splits" not in model.model.get_params()
    assert report["n_samples"] == len(y_count)
    assert np.isfinite(model.predict(X)).all()


def test_unknown_estimator_parameter_is_reported():
    """A genuine typo must raise, rather than being silently dropped as unknown kwargs."""
    X, y_count, _, groups = _synthetic_regions()
    model = RegionRegressor(params={"learning_rat": 0.1})
    with pytest.raises(TypeError, match="learning_rat"):
        model.fit(X, y_count, groups=groups)


def test_grouped_folds_never_split_a_specimen():
    """Preserved leakage protection: folds partition specimens, not individual regions."""
    _, _, _, groups = _synthetic_regions(n_specimens=4, per_specimen=10)
    model = RegionRegressor()
    folds = model._group_folds(groups, 4)

    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert not (set(groups[train_idx]) & set(groups[test_idx])), "specimen leaked across folds"
        assert len(test_idx) > 0
    # Every region is held out exactly once.
    held_out = np.concatenate([test for _, test in folds])
    assert sorted(held_out.tolist()) == list(range(len(groups)))


def test_fit_predictors_fits_both_models_with_one_fold_protocol():
    """Both learned criteria are fitted under identical params and identical CV folds."""
    n_specimens, per_specimen = 4, 10
    _, y_count, y_loss, groups = _synthetic_regions(n_specimens, per_specimen)
    X, _, _, _ = _synthetic_regions(n_specimens, per_specimen)
    region_ids = [f"r{i}" for i in range(len(groups))]

    features = {
        rid: {name: float(X[i, j]) for j, name in enumerate(FEATURE_NAMES)}
        for i, rid in enumerate(region_ids)
    }

    class _Region:
        def __init__(self, n_positives: int):
            self.n_positives = n_positives

    regions_by_id = {rid: _Region(int(abs(y_count[i]))) for i, rid in enumerate(region_ids)}
    # counts are chosen so |count - n_positives| reproduces the intended count-error target
    counts = {rid: 0 for rid in region_ids}
    losses = {rid: float(y_loss[i]) for i, rid in enumerate(region_ids)}

    count_model, loss_model, report = fit_predictors(
        features,
        region_ids,
        counts,
        regions_by_id,
        losses,
        groups=list(groups),
        params=_config_block(),
    )

    # Same hyperparameters on both models, and no CV key among them.
    assert count_model.params == loss_model.params
    assert not (set(count_model.params) & CV_ONLY_PARAMS)
    assert "n_splits" not in count_model.model.get_params()
    assert "n_splits" not in loss_model.model.get_params()

    # The protocol is reported, not guessed: one fold per specimen.
    assert report["n_splits"] == n_specimens
    assert report["n_specimens"] == n_specimens
    assert report["regressor_params"] == count_model.params

    # Out-of-fold scoring covered every region in both criteria.
    assert report["count_error_target"]["oof_n_scored"] == len(region_ids)
    assert report["learned_loss_target"]["oof_n_scored"] == len(region_ids)

    X_rows = feature_matrix([features[r] for r in region_ids])
    assert np.isfinite(count_model.predict(X_rows)).all()
    assert np.isfinite(loss_model.predict(X_rows)).all()


def test_explicit_fold_override_beats_config():
    """Smoke runs may shrink the fold count without touching the shipped config."""
    _, y_count, _, groups = _synthetic_regions(n_specimens=4, per_specimen=10)
    X = np.random.default_rng(1).normal(size=(len(groups), len(FEATURE_NAMES)))
    model = RegionRegressor(params=_config_block())
    report = model.fit(X, y_count, groups=groups, n_splits=2)
    assert report["oof_n_scored"] == len(y_count)
