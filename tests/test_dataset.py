"""Augmentation geometry, label rules and sampling-ratio tests."""

from __future__ import annotations

import numpy as np
import pytest

from midogcx.data.dataset import (
    TrainingItem,
    _clip_boxes,
    positive_enriched_indices,
    transform_boxes,
)


def test_hflip_mirrors_boxes():
    boxes, w, h = transform_boxes(np.array([[10.0, 20.0, 30.0, 40.0]]), 100, 50, True, False, 0)
    assert (w, h) == (100, 50)
    assert boxes[0].tolist() == [70.0, 20.0, 90.0, 40.0]


def test_vflip_mirrors_boxes():
    boxes, _w, _h = transform_boxes(np.array([[10.0, 20.0, 30.0, 40.0]]), 100, 50, False, True, 0)
    assert boxes[0].tolist() == [10.0, 10.0, 30.0, 30.0]


def test_rot90_maps_points_and_swaps_canvas():
    boxes, w, h = transform_boxes(np.array([[10.0, 20.0, 30.0, 40.0]]), 100, 50, False, False, 1)
    assert (w, h) == (50, 100)
    # (x, y) -> (y, W - x): corners (10,20)->(20,90) and (30,40)->(40,70)
    assert boxes[0].tolist() == [20.0, 70.0, 40.0, 90.0]


def test_four_rotations_are_identity():
    boxes, w, h = transform_boxes(np.array([[10.0, 20.0, 30.0, 40.0]]), 100, 50, False, False, 4)
    assert (w, h) == (100, 50)
    assert np.allclose(boxes[0], [10.0, 20.0, 30.0, 40.0])


def test_box_centres_follow_the_same_transform_as_points():
    rng = np.random.default_rng(0)
    boxes = np.array([[10.0, 20.0, 30.0, 40.0], [60.0, 5.0, 80.0, 25.0]])
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    out, w, h = transform_boxes(boxes, 100, 50, True, True, 1)
    out_centres = np.stack([(out[:, 0] + out[:, 2]) / 2, (out[:, 1] + out[:, 3]) / 2], axis=1)

    # Apply the same operations to the centres manually.
    expected = centres.copy()
    expected[:, 0] = expected[:, 1]
    expected[:, 1] = 100 - centres[:, 0]  # rot90 with original width 100
    expected[:, 0] = 50 - expected[:, 0]  # hflip with new width 50
    expected[:, 1] = 100 - expected[:, 1]  # vflip with new height 100
    assert np.allclose(out_centres, expected)
    assert (w, h) == (50, 100)


def test_clip_boxes_drops_degenerate_and_clips():
    # Third box is 0.5 px wide -> degenerate; second is clipped to a 1 px sliver and also
    # dropped because it cannot support a meaningful detection target.
    boxes = np.array([
        [-5.0, -5.0, 20.0, 20.0],
        [99.0, 99.0, 130.0, 130.0],
        [10.0, 10.0, 10.5, 10.5],
        [50.0, 50.0, 70.0, 70.0],
    ])
    out = _clip_boxes(boxes, 100, 100)
    assert out.shape == (2, 4)
    assert out[0].tolist() == [0.0, 0.0, 20.0, 20.0]
    assert out[1].tolist() == [50.0, 50.0, 70.0, 70.0]


def _items(n_pos: int, n_neg: int) -> list[TrainingItem]:
    items = [
        TrainingItem(region_id=f"p{i}", slide_id=1, x=0, y=0, width=512, height=512, has_positive=True)
        for i in range(n_pos)
    ]
    items += [
        TrainingItem(region_id=f"n{i}", slide_id=1, x=0, y=0, width=512, height=512, has_positive=False)
        for i in range(n_neg)
    ]
    return items


def test_positive_enriched_sampling_respects_ratio():
    items = _items(4, 36)
    rng = np.random.default_rng(0)
    idx = positive_enriched_indices(items, n_samples=100, fraction_positive=0.5, rng=rng)
    assert len(idx) == 100
    n_pos = sum(1 for i in idx if items[i].has_positive)
    assert 45 <= n_pos <= 55


def test_positive_enriched_sampling_is_reproducible():
    items = _items(4, 36)
    a = positive_enriched_indices(items, 50, 0.5, np.random.default_rng(7))
    b = positive_enriched_indices(items, 50, 0.5, np.random.default_rng(7))
    assert a == b


def test_sampling_without_positives_falls_back_to_random():
    items = _items(0, 5)
    idx = positive_enriched_indices(items, 10, 0.5, np.random.default_rng(0))
    assert len(idx) == 10


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        positive_enriched_indices(_items(1, 1), 10, 1.5, np.random.default_rng(0))
