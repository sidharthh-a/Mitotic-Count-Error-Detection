"""One-to-one point matching in physical units.

MIDOG++ scores detection with a 7.5 um radius: a predicted point is a true positive only
if it is matched to a distinct ground-truth point within 7.5 um. Matching is one-to-one,
so duplicates remain false positives after suppression, and a prediction cannot "use up"
two ground-truth cells.

Distances are computed after converting pixels to micrometres with the specimen's own
spacing, so anisotropic acquisitions are handled correctly. Predicted and ground-truth
scores are irrelevant here; the caller fixes the score threshold first.

The algorithm is greedy global-nearest: all candidate pairs inside the radius are sorted
by distance and accepted when both endpoints are still free. This is deterministic and
reproduces the standard greedy behaviour; it is *not* proven identical to the official
``evalutils`` implementation on crowded, ambiguous cases (see docs/EVALUATION_PROTOCOL.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchResult:
    n_pred: int
    n_gt: int
    true_positives: int
    false_positives: int
    false_negatives: int
    #: ``(pred_index, gt_index, distance_um)`` for each accepted pair.
    matches: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)

    @property
    def count_error(self) -> int:
        """Absolute counting error; identical to ``|FP - FN|`` under this matching."""
        return abs(self.n_pred - self.n_gt)

    @property
    def signed_bias(self) -> int:
        return self.n_pred - self.n_gt

    @property
    def mean_match_distance_um(self) -> float | None:
        if not self.matches:
            return None
        return float(np.mean([m[2] for m in self.matches]))


def match_points(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    radius_um: float = 7.5,
    spacing_um: tuple[float, float] = (1.0, 1.0),
) -> MatchResult:
    """Match predicted centres to ground-truth centres within ``radius_um``.

    ``spacing_um`` is ``(sx, sy)`` micrometres per pixel. Pass ``(1, 1)`` if the inputs
    are already in micrometres.
    """
    pred = np.asarray(pred_xy, dtype=float)
    gt = np.asarray(gt_xy, dtype=float)
    pred = pred.reshape(-1, 2) if pred.size else np.zeros((0, 2), dtype=float)
    gt = gt.reshape(-1, 2) if gt.size else np.zeros((0, 2), dtype=float)

    sx, sy = float(spacing_um[0]), float(spacing_um[1])
    if sx <= 0 or sy <= 0:
        raise ValueError(f"spacing must be positive, got {spacing_um}")

    scale = np.array([sx, sy], dtype=float)
    pred_um, gt_um = pred * scale, gt * scale

    n_pred, n_gt = len(pred_um), len(gt_um)
    if n_pred == 0 or n_gt == 0:
        return MatchResult(
            n_pred=n_pred,
            n_gt=n_gt,
            true_positives=0,
            false_positives=n_pred,
            false_negatives=n_gt,
            matches=[],
            unmatched_pred=list(range(n_pred)),
            unmatched_gt=list(range(n_gt)),
        )

    from scipy.spatial import cKDTree

    tree = cKDTree(gt_um)
    candidates: list[tuple[float, int, int]] = []
    for i, point in enumerate(pred_um):
        for j in tree.query_ball_point(point, r=radius_um):
            dist = float(np.linalg.norm(point - gt_um[j]))
            if dist <= radius_um:
                candidates.append((dist, i, int(j)))

    # Deterministic ordering: nearest first, then prediction index, then GT index.
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for dist, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matches.append((i, j, dist))

    tp = len(matches)
    return MatchResult(
        n_pred=n_pred,
        n_gt=n_gt,
        true_positives=tp,
        false_positives=n_pred - tp,
        false_negatives=n_gt - tp,
        matches=matches,
        unmatched_pred=[i for i in range(n_pred) if i not in used_pred],
        unmatched_gt=[j for j in range(n_gt) if j not in used_gt],
    )
