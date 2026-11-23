"""Duplicate-suppression tests."""

from __future__ import annotations

import numpy as np
import pytest

from midogcx.evaluation.nms import nms_by_distance, nms_keep


def test_cluster_keeps_highest_scoring_point():
    points = np.array([[0.0, 0.0], [5.0, 0.0], [60.0, 0.0]])
    scores = np.array([0.4, 0.9, 0.8])
    keep = nms_by_distance(points, scores, radius=25)
    assert sorted(keep.tolist()) == [1, 2]


def test_points_outside_radius_survive():
    points = np.array([[0.0, 0.0], [26.0, 0.0]])
    keep = nms_by_distance(points, np.array([0.9, 0.8]), radius=25)
    assert sorted(keep.tolist()) == [0, 1]


def test_radius_boundary_is_inclusive():
    points = np.array([[0.0, 0.0], [25.0, 0.0]])
    keep = nms_by_distance(points, np.array([0.9, 0.8]), radius=25)
    assert keep.tolist() == [0]


def test_ties_are_broken_by_index_deterministically():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    scores = np.array([0.5, 0.5])
    keep = nms_by_distance(points, scores, radius=25)
    assert keep.tolist() == [0]


def test_empty_input():
    assert nms_by_distance(np.zeros((0, 2)), np.zeros(0), radius=25).size == 0


def test_threshold_is_strict():
    points = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    scores = np.array([0.5, 0.51, 0.49])
    kept_points, kept_scores = nms_keep(points, scores, radius=25, threshold=0.5)
    # score == 0.5 is excluded, matching the official `score > threshold` rule.
    assert kept_scores.tolist() == pytest.approx([0.51])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        nms_by_distance(np.zeros((2, 2)), np.zeros(3), radius=25)
