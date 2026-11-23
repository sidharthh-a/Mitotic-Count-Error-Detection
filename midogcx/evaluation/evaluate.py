"""Turning raw detector output into scoring results.

Pipeline per specimen (order matters and is fixed for every strategy):

1. apply a strict ``score > threshold`` filter (matches the official evaluator),
2. suppress duplicates across the whole specimen by centre distance,
3. match the surviving centres to ground-truth centres one-to-one within 7.5 um,
4. record TP/FP/FN, counts, count error and signed bias.

Thresholds are selected on ``source_calib`` specimens only. Target assessment specimens
are never used to choose a threshold, a strategy or a hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from ..utils import load_json, save_json
from .matching import MatchResult, match_points
from .metrics import SpecimenMetrics, aggregate_metrics
from .nms import nms_keep


@dataclass
class SlidePrediction:
    slide_id: int
    points: np.ndarray  # (N, 2) pixel centres
    scores: np.ndarray  # (N,)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 2)
        self.scores = np.asarray(self.scores, dtype=float).reshape(-1)
        if len(self.points) != len(self.scores):
            raise ValueError("points and scores must have the same length")

    def to_dict(self) -> dict:
        return {
            "slide_id": int(self.slide_id),
            "points": self.points.tolist(),
            "scores": self.scores.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SlidePrediction":
        return cls(
            slide_id=int(payload["slide_id"]),
            points=np.asarray(payload.get("points", []), dtype=float),
            scores=np.asarray(payload.get("scores", []), dtype=float),
        )


def save_predictions(predictions: Iterable[SlidePrediction], path) -> None:
    """Persist raw predictions so any threshold can be reapplied without re-inference."""
    save_json({"predictions": [p.to_dict() for p in predictions]}, path)


def load_predictions(path) -> dict[int, SlidePrediction]:
    payload = load_json(path)
    return {int(p["slide_id"]): SlidePrediction.from_dict(p) for p in payload["predictions"]}


def evaluate_slide(
    prediction: SlidePrediction,
    gt_xy: np.ndarray,
    spacing_um: tuple[float, float],
    score_threshold: float,
    nms_radius_px: float,
    match_radius_um: float = 7.5,
) -> tuple[SpecimenMetrics, MatchResult]:
    """Score one specimen and return both the metrics row and the raw match detail."""
    kept_points, kept_scores = nms_keep(
        prediction.points, prediction.scores, radius=nms_radius_px, threshold=score_threshold
    )
    result = match_points(
        kept_points, np.asarray(gt_xy, dtype=float), radius_um=match_radius_um, spacing_um=spacing_um
    )
    metrics = SpecimenMetrics(
        slide_id=int(prediction.slide_id),
        n_pred=result.n_pred,
        n_gt=result.n_gt,
        true_positives=result.true_positives,
        false_positives=result.false_positives,
        false_negatives=result.false_negatives,
        mean_match_distance_um=result.mean_match_distance_um,
    )
    return metrics, result


def evaluate_specimens(
    predictions: dict[int, SlidePrediction],
    slide_ids: Iterable[int],
    gt_fn: Callable[[int], np.ndarray],
    spacing: dict[int, tuple[float, float]],
    score_threshold: float,
    nms_radius_px: float,
    match_radius_um: float = 7.5,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Aggregate metrics over a fixed specimen list.

    A missing prediction for a listed specimen raises: silence there would silently drop a
    hard specimen from the average.
    """
    results: list[SpecimenMetrics] = []
    for sid in slide_ids:
        sid = int(sid)
        if sid not in predictions:
            raise KeyError(f"no prediction for specimen {sid}")
        metrics, _ = evaluate_slide(
            predictions[sid],
            gt_fn(sid),
            spacing_um=spacing[sid],
            score_threshold=score_threshold,
            nms_radius_px=nms_radius_px,
            match_radius_um=match_radius_um,
        )
        results.append(metrics)
    summary = aggregate_metrics(results, n_boot=n_boot, seed=seed)
    summary["score_threshold"] = float(score_threshold)
    summary["nms_radius_px"] = float(nms_radius_px)
    summary["match_radius_um"] = float(match_radius_um)
    return summary


def select_threshold(
    predictions: dict[int, SlidePrediction],
    slide_ids: Iterable[int],
    gt_fn: Callable[[int], np.ndarray],
    spacing: dict[int, tuple[float, float]],
    thresholds: Iterable[float],
    nms_radius_px: float,
    match_radius_um: float = 7.5,
) -> dict:
    """Pick the pooled-F1 maximising threshold on calibration specimens.

    Mirrors the official evaluator, which searches ``0.50 .. 0.99`` and reports the
    threshold that maximises aggregate F1 on the validation set. The search range is
    recorded in the output so it is never confused with an official sweep.

    Tie-break: pooled F1 is discrete, so exact ties are common. The **higher** threshold
    wins, i.e. the tie is settled towards fewer and more confident detections. The
    official evaluator's ``argmax`` would keep the first (lowest) maximum instead. This
    is a deliberate, recorded deviation (docs/EVALUATION_PROTOCOL.md), not an accident;
    the full sweep is saved so a reader can apply the other convention offline.
    """
    thresholds = [float(t) for t in thresholds]
    rows = []
    for thr in thresholds:
        summary = evaluate_specimens(
            predictions,
            slide_ids,
            gt_fn=gt_fn,
            spacing=spacing,
            score_threshold=thr,
            nms_radius_px=nms_radius_px,
            match_radius_um=match_radius_um,
            n_boot=0,
        )
        rows.append(
            {
                "threshold": thr,
                "f1": summary["pooled"]["f1"],
                "precision": summary["pooled"]["precision"],
                "recall": summary["pooled"]["recall"],
            }
        )
    best = max(rows, key=lambda r: (r["f1"], r["threshold"]))
    return {
        "best_threshold": best["threshold"],
        "search_range": [min(thresholds), max(thresholds)],
        "criterion": "pooled F1 on the calibration specimens",
        "sweep": rows,
    }
