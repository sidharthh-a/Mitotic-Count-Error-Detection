"""Point-matching tests, mirroring the eight synthetic cases previously checked against
evalutils 0.3.1, plus anisotropy and tie-determinism cases."""

from __future__ import annotations

import numpy as np
import pytest

from midogcx.evaluation.matching import match_points


def _pairs(xy):
    return np.asarray(xy, dtype=float).reshape(-1, 2)


def test_exact_match_is_true_positive():
    r = match_points(_pairs([[10, 10]]), _pairs([[10, 10]]), radius_um=7.5, spacing_um=(0.25, 0.25))
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 0, 0)
    assert r.matches[0][2] == pytest.approx(0.0)


def test_just_inside_radius_matches():
    # radius 7.5 um at 0.25 um/px = 30 px. Use 29 px offset.
    r = match_points(_pairs([[0, 0]]), _pairs([[29, 0]]), radius_um=7.5, spacing_um=(0.25, 0.25))
    assert r.true_positives == 1
    assert r.matches[0][2] == pytest.approx(29 * 0.25)


def test_just_outside_radius_does_not_match():
    # 31 px * 0.25 = 7.75 um > 7.5 um
    r = match_points(_pairs([[0, 0]]), _pairs([[31, 0]]), radius_um=7.5, spacing_um=(0.25, 0.25))
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 1, 1)


def test_boundary_is_inclusive():
    # Exactly 7.5 um (30 px) is inside, per the <= radius convention.
    r = match_points(_pairs([[0, 0]]), _pairs([[30, 0]]), radius_um=7.5, spacing_um=(0.25, 0.25))
    assert r.true_positives == 1


def test_duplicate_predictions_become_false_positives():
    preds = _pairs([[10, 10], [11, 11]])
    gts = _pairs([[10, 10]])
    r = match_points(preds, gts, radius_um=7.5, spacing_um=(0.25, 0.25))
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 1, 0)
    # The nearest prediction must be the one kept as the true positive.
    assert r.matches[0][0] == 0


def test_two_targets_and_one_prediction():
    preds = _pairs([[0, 0]])
    gts = _pairs([[1, 0], [2, 0]])
    r = match_points(preds, gts, radius_um=7.5, spacing_um=(0.25, 0.25))
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 0, 1)
    assert r.matches[0][1] == 0  # nearer ground truth


def test_empty_ground_truth_all_false_positives():
    r = match_points(_pairs([[5, 5]]), _pairs([]), radius_um=7.5, spacing_um=(0.25, 0.25))
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 1, 0)


def test_empty_predictions_all_false_negatives():
    r = match_points(_pairs([]), _pairs([[5, 5]]), radius_um=7.5, spacing_um=(0.25, 0.25))
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 0, 1)


def test_both_empty_is_zero_everything():
    r = match_points(_pairs([]), _pairs([]), radius_um=7.5, spacing_um=(0.25, 0.25))
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 0, 0)
    assert r.mean_match_distance_um is None


def test_anisotropic_spacing_changes_the_verdict():
    # 60 px apart in y only. With 0.125 um/px vertically that is 7.5 um (match);
    # with 0.25 um/px it is 15 um (no match).
    preds, gts = _pairs([[0, 0]]), _pairs([[0, 60]])
    assert match_points(preds, gts, 7.5, (0.25, 0.125)).true_positives == 1
    assert match_points(preds, gts, 7.5, (0.25, 0.25)).true_positives == 0


def test_greedy_assignment_is_deterministic_and_one_to_one():
    rng = np.random.default_rng(7)
    preds = rng.uniform(0, 100, size=(200, 2))
    gts = rng.uniform(0, 100, size=(50, 2))
    a = match_points(preds, gts, 7.5, (0.25, 0.25))
    b = match_points(preds, gts, 7.5, (0.25, 0.25))
    assert a.matches == b.matches
    # One-to-one: no prediction and no ground truth is used twice.
    assert len({m[0] for m in a.matches}) == len(a.matches)
    assert len({m[1] for m in a.matches}) == len(a.matches)
    assert a.true_positives == len(a.matches)
    assert a.false_positives == a.n_pred - a.true_positives
    assert a.false_negatives == a.n_gt - a.true_positives


def test_count_bookkeeping_identity():
    # The blueprint's cancellation point: |FP - FN| equals the absolute count error.
    r = match_points(
        np.asarray([[0, 0]] * 3, dtype=float),
        np.asarray([[0, 0]], dtype=float),
        7.5,
        (0.25, 0.25),
    )
    assert r.count_error == abs(r.false_positives - r.false_negatives)
    assert r.signed_bias == r.false_positives - r.false_negatives
