"""Duplicate suppression by centre distance.

MIDOG++ suppresses duplicates across a whole specimen (not per tile) using a greedy
centre-distance rule with a KD-tree; this mirrors that behaviour so tile seams cannot
produce duplicate detections. The radius is configurable because it is expressed in
pixels while the physical radius is fixed in micrometres.

Determinism: ties in score are broken by ascending original index, so the same input
always yields the same kept set.
"""

from __future__ import annotations

import numpy as np


def nms_by_distance(
    points: np.ndarray,
    scores: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return indices kept by greedy centre-distance suppression.

    ``points`` is ``(N, 2)`` in pixels, ``scores`` is ``(N,)``. Points are visited in
    descending score order and any not-yet-kept point within ``radius`` pixels of a kept
    point is removed. Because suppression is applied to kept points only, the result
    differs slightly from a fixed-order sweep but matches the published implementation's
    intent: the highest-scoring detection in a cluster survives.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=int)
    if len(points) != n:
        raise ValueError(f"points/scores length mismatch: {len(points)} vs {n}")
    if radius <= 0:
        return np.argsort(-scores, kind="stable")

    from scipy.spatial import cKDTree

    order = np.lexsort((np.arange(n), -scores))  # primary: -score, secondary: index
    tree = cKDTree(points)
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        for neighbour in tree.query_ball_point(points[idx], r=radius):
            suppressed[neighbour] = True
    return np.asarray(keep, dtype=int)


def nms_keep(
    points: np.ndarray,
    scores: np.ndarray,
    radius: float,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an optional strict ``score > threshold`` filter, then distance NMS."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if threshold is not None:
        mask = scores > threshold  # strict, matching the official evaluator
        points, scores = points[mask], scores[mask]
    keep = nms_by_distance(points, scores, radius)
    return points[keep], scores[keep]
