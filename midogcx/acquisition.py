"""Acquisition pipeline pieces: counts, feature rows, predictor fitting.

Kept separate from the CLI so the leakage-relevant logic is unit-testable:

* features are built from detector outputs and geometry only;
* true counts and region losses are attached **after** the features exist, and only for
  specimens in the error-training cohort;
* ``fit_predictors`` cross-fits by specimen and reports out-of-fold quality next to the
  in-sample fit, so an optimistic number cannot be mistaken for a validated one.
"""

from __future__ import annotations

import numpy as np

from .data.patches import Region
from .evaluation.nms import nms_keep
from .models.error_predictor import (
    FEATURE_NAMES,
    RegionRegressor,
    region_features,
)


def count_above_threshold(
    points: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    nms_radius_px: float = 25.0,
) -> dict:
    """Locked threshold + duplicate suppression, then count. Returns counts and kept data."""
    kept_points, kept_scores = nms_keep(points, scores, radius=nms_radius_px, threshold=threshold)
    return {"n_above": int(len(kept_scores)), "points": kept_points, "scores": kept_scores}


def build_feature_rows(
    regions: list[Region],
    detections: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold: float,
    nms_radius_px: float = 25.0,
    tissue_fraction: dict[str, float] | None = None,
) -> tuple[dict[str, dict], dict[str, int]]:
    """Feature rows and locked counts for each region. No ground truth is used."""
    features: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for region in regions:
        points, scores = detections.get(region.region_id, (np.zeros((0, 2)), np.zeros(0)))
        counted = count_above_threshold(points, scores, threshold, nms_radius_px)
        counts[region.region_id] = counted["n_above"]
        features[region.region_id] = region_features(
            region,
            np.asarray(points, dtype=float),
            np.asarray(scores, dtype=float),
            threshold=threshold,
            tissue_fraction=(tissue_fraction or {}).get(region.region_id, float("nan")),
        )
    return features, counts


def attach_targets(
    region_ids: list[str],
    counts: dict[str, int],
    regions_by_id: dict[str, Region],
) -> np.ndarray:
    """Supervision for the count-error learner: ``|predicted - true|`` per region.

    Called only for specimens whose annotations are allowed to be revealed. The selector
    never receives these values for target-pool regions.
    """
    return np.asarray(
        [abs(counts[rid] - regions_by_id[rid].n_positives) for rid in region_ids], dtype=float
    )


def fit_predictors(
    features: dict[str, dict],
    region_ids: list[str],
    counts: dict[str, int],
    regions_by_id: dict[str, Region],
    region_losses: dict[str, float],
    groups: list[int],
    params: dict | None = None,
    n_splits: int | None = None,
) -> tuple[RegionRegressor, RegionRegressor, dict]:
    """Fit the count-error and learned-loss predictors under identical conditions.

    ``params`` is the shared predictor config block. Cross-validation settings it contains
    (``n_splits``) are peeled off here and never forwarded to the estimator, so both models
    receive exactly the same hyperparameters and exactly the same fold count.
    """
    from .models.error_predictor import feature_matrix, split_cv_params

    estimator_params, n_splits = split_cv_params(params, n_splits)

    X = feature_matrix([features[rid] for rid in region_ids], FEATURE_NAMES)
    y_count = attach_targets(region_ids, counts, regions_by_id)
    y_loss = np.asarray([region_losses[rid] for rid in region_ids], dtype=float)
    group_arr = np.asarray(groups)

    count_model = RegionRegressor(params=estimator_params, target_name="abs_count_error")
    loss_model = RegionRegressor(params=estimator_params, target_name="detector_loss")
    count_report = count_model.fit(X, y_count, groups=group_arr, n_splits=n_splits)
    loss_report = loss_model.fit(X, y_loss, groups=group_arr, n_splits=n_splits)

    report = {
        "n_regions": len(region_ids),
        "n_specimens": int(len(set(groups))),
        # Effective fold count, as reported by the fit: GroupKFold caps it at the number of
        # specimens, which for this cohort (10 specimens, 5 fitted folds) matters.
        "n_splits": int(count_report.get("n_splits", n_splits)),
        "regressor_params": estimator_params,
        "count_error_target": {
            "mean": float(y_count.mean()),
            "median": float(np.median(y_count)),
            "fraction_zero": float((y_count == 0).mean()),
            **count_report,
        },
        "learned_loss_target": loss_report,
        "leakage_note": (
            "Targets use out-of-sample predictions from a source detector that never saw "
            "these specimens in fitting. Feature rows and targets are never built for "
            "target-pool regions."
        ),
    }
    return count_model, loss_model, report


def criterion_agreement(ranked_a: list[tuple[str, float]], ranked_b: list[tuple[str, float]], k: int) -> float:
    from .selection.strategies import ranking_overlap

    return ranking_overlap(ranked_a, ranked_b, k=k)
