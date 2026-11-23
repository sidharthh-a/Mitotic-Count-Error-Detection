"""The five acquisition strategies compared in this study.

Every strategy receives **only** unlabelled information: detector outputs on the target
images, region geometry and the tissue mask. Numbered definitions:

1. ``random`` – uniform seeded permutation of the candidate pool.
2. ``uncertainty`` – total predictive entropy of the region's *borderline* detections,
   ``U(r) = sum_{d in D_r, |s_d - tau| <= delta} h(s_d)`` with ``h`` the binary entropy.
   Detections far from the threshold are not uncertain evidence, so a region with no
   borderline detection scores 0 and ranks last. All candidate regions have the same
   pixel size, so this aggregate is not biased by region length (an explicit concern in
   the protocol); edge regions are slightly smaller and this is reported.
3. ``density`` – predicted mitoses per mm^2 of tissue, ``n_above(tau) / area_mm2``.
4. ``learned_loss`` – a learned predictor of the detector's training loss on the region
   (the generic "predicted loss" criterion of Yoo & Kweon 2019).
5. ``count_error`` – **proposed**: a learned predictor of the absolute counting error
   ``|n_r - y_r|`` the region would exhibit after annotation.

Optional, not part of the primary five: ``scale_disagreement``, ranking by the difference
between counts obtained at two image scales. It is motivated by Xu et al. (2026) but is
*not* a faithful reproduction of their ISBI method and is only used if two-scale
predictions are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STRATEGIES: tuple[str, ...] = (
    "random",
    "uncertainty",
    "density",
    "learned_loss",
    "count_error",
)

OPTIONAL_STRATEGIES: tuple[str, ...] = ("scale_disagreement",)


@dataclass
class StrategyContext:
    """Everything a ranking rule may look at (no ground-truth labels)."""

    detections: dict[str, tuple[np.ndarray, np.ndarray]]
    threshold: float
    seed: int = 0
    uncertainty_delta: float = 0.1
    predictors: dict = field(default_factory=dict)  # {'count_error': RegionRegressor, ...}
    features: dict | None = None  # region_id -> feature dict
    second_scale_counts: dict[str, float] | None = None


def _entropy(scores: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=float), 1e-9, 1 - 1e-9)
    return -s * np.log(s) - (1 - s) * np.log(1 - s)


def rank_random(regions, seed: int = 0) -> list[tuple[str, float]]:
    rng = np.random.default_rng(seed)
    ids = [r.region_id for r in regions]
    order = rng.permutation(len(ids))
    # Score = descending position so the ranking is the permutation itself.
    scores = np.empty(len(ids), dtype=float)
    scores[order] = np.arange(len(ids))[::-1]
    return sorted(zip(ids, scores.tolist()), key=lambda t: (-t[1], t[0]))


def rank_uncertainty(
    regions, detections, threshold: float, delta: float = 0.1
) -> list[tuple[str, float]]:
    out = []
    for region in regions:
        points, scores = detections.get(region.region_id, (np.zeros((0, 2)), np.zeros(0)))
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            out.append((region.region_id, 0.0))
            continue
        band = np.abs(scores - threshold) <= delta
        value = float(_entropy(scores[band]).sum()) if band.any() else 0.0
        out.append((region.region_id, value))
    return sorted(out, key=lambda t: (-t[1], t[0]))


def rank_density(regions, detections, threshold: float) -> list[tuple[str, float]]:
    out = []
    for region in regions:
        _points, scores = detections.get(region.region_id, (np.zeros((0, 2)), np.zeros(0)))
        scores = np.asarray(scores, dtype=float)
        area_mm2 = region.tissue_area_um2 / 1e6
        n_above = int((scores > threshold).sum())
        value = float(n_above / area_mm2) if area_mm2 > 0 else 0.0
        out.append((region.region_id, value))
    return sorted(out, key=lambda t: (-t[1], t[0]))


def _rank_learned(regions, predictor, features) -> list[tuple[str, float]]:
    from ..models.error_predictor import feature_matrix

    if predictor is None or features is None:
        raise ValueError("learned strategy requires a fitted predictor and feature rows")
    ids = [r.region_id for r in regions]
    missing = [rid for rid in ids if rid not in features]
    if missing:
        raise KeyError(f"missing features for {len(missing)} regions, e.g. {missing[:3]}")
    X = feature_matrix([features[rid] for rid in ids], predictor.feature_names)
    preds = predictor.predict(X)
    return sorted(zip(ids, preds.tolist()), key=lambda t: (-t[1], t[0]))


def rank_learned_loss(regions, context: StrategyContext) -> list[tuple[str, float]]:
    return _rank_learned(regions, context.predictors.get("learned_loss"), context.features)


def rank_count_error(regions, context: StrategyContext) -> list[tuple[str, float]]:
    return _rank_learned(regions, context.predictors.get("count_error"), context.features)


def rank_scale_disagreement(regions, context: StrategyContext) -> list[tuple[str, float]]:
    """Absolute count difference between two inference scales (retrospective comparator)."""
    if context.second_scale_counts is None:
        raise ValueError("scale_disagreement needs second_scale_counts")
    out = []
    for region in regions:
        _points, scores = context.detections.get(region.region_id, (np.zeros((0, 2)), np.zeros(0)))
        n_main = float((np.asarray(scores) > context.threshold).sum())
        n_second = float(context.second_scale_counts.get(region.region_id, np.nan))
        value = abs(n_main - n_second) if np.isfinite(n_second) else 0.0
        out.append((region.region_id, value))
    return sorted(out, key=lambda t: (-t[1], t[0]))


def rank_regions(regions, strategy: str, context: StrategyContext) -> list[tuple[str, float]]:
    """Dispatch to a strategy. Unknown names raise instead of silently defaulting."""
    if strategy == "random":
        return rank_random(regions, seed=context.seed)
    if strategy == "uncertainty":
        return rank_uncertainty(
            regions, context.detections, context.threshold, delta=context.uncertainty_delta
        )
    if strategy == "density":
        return rank_density(regions, context.detections, context.threshold)
    if strategy == "learned_loss":
        return rank_learned_loss(regions, context)
    if strategy == "count_error":
        return rank_count_error(regions, context)
    if strategy == "scale_disagreement":
        return rank_scale_disagreement(regions, context)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES + OPTIONAL_STRATEGIES}")


def ranking_overlap(a: list[tuple[str, float]], b: list[tuple[str, float]], k: int = 20) -> float:
    """Fraction of the top-``k`` of ranking ``a`` that also appears in the top-``k`` of ``b``."""
    top_a = {rid for rid, _ in a[:k]}
    top_b = {rid for rid, _ in b[:k]}
    return len(top_a & top_b) / max(1, len(top_a))
