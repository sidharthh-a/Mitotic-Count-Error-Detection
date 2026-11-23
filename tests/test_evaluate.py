"""End-to-end evaluation of saved predictions, including threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from midogcx.evaluation.evaluate import (
    SlidePrediction,
    evaluate_slide,
    evaluate_specimens,
    load_predictions,
    save_predictions,
    select_threshold,
)

SPACING = (0.25, 0.25)


def test_perfect_predictions_score_one():
    gt = np.array([[100.0, 100.0], [300.0, 300.0]])
    pred = SlidePrediction(1, gt.copy(), np.array([0.9, 0.9]))
    metrics, match = evaluate_slide(pred, gt, SPACING, score_threshold=0.5, nms_radius_px=25)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (2, 0, 0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.abs_count_error == 0


def test_clustered_predictions_are_suppressed_before_matching():
    gt = np.array([[100.0, 100.0]])
    pred = SlidePrediction(1, np.array([[100.0, 100.0], [110.0, 100.0], [120.0, 100.0]]), np.array([0.9, 0.8, 0.7]))
    metrics, _ = evaluate_slide(pred, gt, SPACING, score_threshold=0.5, nms_radius_px=25)
    # Whole-image NMS keeps the highest-scoring point of the cluster only.
    assert metrics.n_pred == 1
    assert (metrics.true_positives, metrics.false_positives) == (1, 0)


def test_threshold_filter_is_strict():
    gt = np.array([[100.0, 100.0]])
    pred = SlidePrediction(1, np.array([[100.0, 100.0], [500.0, 500.0]]), np.array([0.5, 0.6]))
    metrics, _ = evaluate_slide(pred, gt, SPACING, score_threshold=0.5, nms_radius_px=25)
    # score == 0.5 excluded; score 0.6 far from GT -> one FP, one FN.
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (0, 1, 1)


def test_counting_cancellation_is_detectable_in_the_result():
    gt = np.array([[100.0, 100.0]])
    preds = np.array([[100.0, 100.0]] + [[2000.0 + i * 100, 100.0] for i in range(5)])
    pred = SlidePrediction(1, preds, np.array([0.9] * 6))
    metrics, _ = evaluate_slide(pred, gt, SPACING, 0.5, 25)
    assert metrics.abs_count_error == 5
    assert metrics.false_positives == 5 and metrics.false_negatives == 0


def test_missing_prediction_raises():
    with pytest.raises(KeyError):
        evaluate_specimens(
            {},
            [1],
            gt_fn=lambda sid: np.zeros((0, 2)),
            spacing={1: SPACING},
            score_threshold=0.5,
            nms_radius_px=25,
            n_boot=0,
        )


def test_evaluate_specimens_aggregates_over_all_listed_specimens():
    gts = {1: np.array([[100.0, 100.0]]), 2: np.array([[200.0, 200.0], [400.0, 400.0]])}
    preds = {
        1: SlidePrediction(1, np.array([[100.0, 100.0]]), np.array([0.9])),
        2: SlidePrediction(2, np.array([[200.0, 200.0]]), np.array([0.9])),
    }
    out = evaluate_specimens(
        preds, [1, 2], gt_fn=lambda s: gts[s], spacing={1: SPACING, 2: SPACING},
        score_threshold=0.5, nms_radius_px=25, n_boot=20,
    )
    assert out["n_specimens"] == 2
    assert out["pooled"]["tp"] == 2 and out["pooled"]["fn"] == 1
    assert out["counting"]["mae"] == pytest.approx(0.5)


def test_threshold_selection_uses_pooled_f1():
    gt = np.array([[100.0, 100.0]])
    pred = SlidePrediction(1, np.array([[100.0, 100.0]]), np.array([0.7]))
    out = select_threshold(
        {1: pred}, [1], gt_fn=lambda s: gt, spacing={1: SPACING},
        thresholds=[0.5, 0.6, 0.7, 0.8], nms_radius_px=25,
    )
    assert out["best_threshold"] in (0.5, 0.6)  # both keep the TP and score F1 = 1
    assert out["search_range"] == [0.5, 0.8]
    assert len(out["sweep"]) == 4


def test_prediction_roundtrip(tmp_path):
    preds = [SlidePrediction(1, np.array([[1.0, 2.0]]), np.array([0.42]))]
    path = tmp_path / "preds.json"
    save_predictions(preds, path)
    loaded = load_predictions(path)
    assert loaded[1].points.tolist() == [[1.0, 2.0]]
    assert loaded[1].scores.tolist() == pytest.approx([0.42])


def test_prediction_validates_lengths():
    with pytest.raises(ValueError):
        SlidePrediction(1, np.zeros((2, 2)), np.zeros(3))
