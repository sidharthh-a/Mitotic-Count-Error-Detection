"""Region-level feature extraction and learned ranking models.

Two learned acquisition rules live here:

* ``count_error``: predict the **absolute mitotic counting error** ``|n_r - y_r|`` of a
  region, where ``n_r`` is the detector's count under the locked threshold and ``y_r`` the
  true count. This is the project's proposed criterion.
* ``learned_loss``: predict the detector's **training loss** on the region (the generic
  predicted-loss criterion of Yoo & Kweon). It exists so the proposed criterion is
  compared against the generic one under identical supervision, not assumed better.

Both are trained by ``RegionRegressor`` on the same source-domain regions, with the same
out-of-fold protocol, the same feature matrix and the same regressor family. That is what
makes the comparison a comparison of *targets* rather than of capacity.

Leakage control: training targets come from out-of-fold detector predictions
(``GroupKFold`` by specimen) so the features are never produced by a detector that saw
the same specimen in fitting. The selector at acquisition time receives only feature rows
built from unlabelled target images.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.patches import Region

#: Config keys that configure cross-validation rather than the estimator. They are allowed
#: to share a block with the hyperparameters for readability, but they are arguments of
#: ``fit()`` (GroupKFold fold count), never keywords of the estimator: sklearn raises
#: ``TypeError`` if one is forwarded.
CV_ONLY_PARAMS: frozenset[str] = frozenset({"n_splits"})

DEFAULT_N_SPLITS: int = 5


#: Feature order is fixed and persisted with the model; adding a feature invalidates
#: previously fitted predictors, which is preferable to a silent column shift.
FEATURE_NAMES: tuple[str, ...] = (
    "n_det_floor",
    "n_det_above_threshold",
    "det_per_mm2",
    "score_max",
    "score_mean",
    "score_std",
    "score_sum_above_threshold",
    "n_borderline",
    "mean_nearest_neighbour_px",
    "tissue_area_mm2",
    "tissue_fraction",
)


def _nearest_neighbour_px(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("nan")
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(points).query(points, k=2)
    return float(np.mean(dist[:, 1]))


def region_features(
    region: Region,
    points: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    tissue_fraction: float = float("nan"),
    borderline_delta: float = 0.1,
) -> dict:
    """Feature row for one region from its raw detections.

    All inputs are available before annotation: detector outputs, region geometry and the
    tissue mask. Ground-truth counts are deliberately absent.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    area_mm2 = region.tissue_area_um2 / 1e6
    above = scores > threshold
    n_floor = int(len(scores))
    n_above = int(above.sum())

    if n_floor:
        score_max = float(scores.max())
        score_mean = float(scores.mean())
        score_std = float(scores.std())
        nn_px = _nearest_neighbour_px(points)
    else:
        score_max = 0.0
        score_mean = 0.0
        score_std = 0.0
        nn_px = float("nan")

    return {
        "n_det_floor": float(n_floor),
        "n_det_above_threshold": float(n_above),
        "det_per_mm2": float(n_above / area_mm2) if area_mm2 > 0 else 0.0,
        "score_max": score_max,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_sum_above_threshold": float(scores[above].sum()) if n_above else 0.0,
        "n_borderline": float(np.sum(np.abs(scores - threshold) <= borderline_delta)),
        "mean_nearest_neighbour_px": nn_px,
        "tissue_area_mm2": float(area_mm2),
        "tissue_fraction": float(tissue_fraction),
    }


def feature_matrix(rows: list[dict], names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    return np.asarray([[float(r.get(n, np.nan)) for n in names] for r in rows], dtype=float)


def split_cv_params(
    params: dict | None, n_splits: int | None = None
) -> tuple[dict, int]:
    """Separate estimator hyperparameters from cross-validation settings.

    A predictor config block carries both (``configs/base.yaml`` keeps ``n_splits`` next to
    the boosting hyperparameters). Only the hyperparameters may reach
    ``HistGradientBoostingRegressor``; the fold count belongs to ``GroupKFold``.

    An explicit ``n_splits`` argument wins over the config value, so callers can override
    the protocol for a smoke run without editing the configuration.
    """
    params = dict(params or {})
    estimator_params = {k: v for k, v in params.items() if k not in CV_ONLY_PARAMS}
    if n_splits is None:
        configured = params.get("n_splits")
        n_splits = DEFAULT_N_SPLITS if configured is None else int(configured)
    return estimator_params, int(n_splits)


@dataclass
class RegionRegressor:
    """Small gradient-boosted regressor over region features."""

    feature_names: tuple[str, ...] = FEATURE_NAMES
    params: dict = field(default_factory=dict)
    model: object | None = None
    target_name: str = "target"
    #: Contains the effective fold count under ``n_splits`` once fitted; persisted with the
    #: model so a reloaded predictor still describes the protocol it was validated under.
    fit_report: dict = field(default_factory=dict)

    def _new_model(self):
        from inspect import signature

        from sklearn.ensemble import HistGradientBoostingRegressor

        params = {
            "max_depth": 3,
            "max_iter": 200,
            "learning_rate": 0.05,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": 0,
        }
        # Defence in depth: even if a CV setting arrives here despite split_cv_params(),
        # it must not reach the estimator. Anything else unknown is a real mistake (typo,
        # renamed sklearn argument) and is reported instead of silently ignored.
        params.update({k: v for k, v in self.params.items() if k not in CV_ONLY_PARAMS})
        unexpected = sorted(set(params) - set(signature(HistGradientBoostingRegressor).parameters))
        if unexpected:
            raise TypeError(
                f"unsupported regressor parameter(s) {unexpected}; cross-validation settings "
                f"such as {sorted(CV_ONLY_PARAMS)} are arguments of fit(), not of the estimator"
            )
        return HistGradientBoostingRegressor(**params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        n_splits: int | None = None,
    ) -> dict:
        """Fit on all data and, when ``groups`` are given, report out-of-fold quality.

        ``n_splits`` defaults to the configured value (``self.params['n_splits']``, else
        ``DEFAULT_N_SPLITS``) and is used only for grouping folds.
        """
        _, n_splits = split_cv_params(self.params, n_splits)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        if len(y) < 10:
            raise ValueError(f"need at least 10 samples to fit, got {len(y)}")

        report: dict = {"n_samples": int(len(y)), "target": self.target_name}

        if groups is not None:
            folds = self._group_folds(np.asarray(groups), n_splits)
            oof = np.full(len(y), np.nan)
            for train_idx, test_idx in folds:
                m = self._new_model()
                m.fit(X[train_idx], y[train_idx])
                oof[test_idx] = m.predict(X[test_idx])
            # Record the *effective* fold count: GroupKFold caps k at the number of groups,
            # so a configured 5 folds on 4 specimens really runs 4. Reporting the requested
            # value would misdescribe the protocol in predictor_report.json.
            report["n_splits"] = len(folds)
            report["n_groups"] = int(len(np.unique(groups)))
            report.update(self._quality(y, oof, prefix="oof_"))
            # Also report in-sample fit so the gap is visible rather than implied.
            in_sample_model = self._new_model().fit(X, y)
            report.update(self._quality(y, in_sample_model.predict(X), prefix="insample_"))
            self.fit_report = report

        self.model = self._new_model().fit(X, y)
        self.fit_report = report
        return report

    def _group_folds(self, groups: np.ndarray, n_splits: int):
        from sklearn.model_selection import GroupKFold

        unique = np.unique(groups)
        k = min(n_splits, len(unique))
        if k < 2:
            raise ValueError("need at least two groups for out-of-fold fitting")
        return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))

    @staticmethod
    def _quality(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
        from scipy.stats import spearmanr

        mask = np.isfinite(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        mae = float(np.mean(np.abs(yt - yp))) if len(yt) else float("nan")
        corr = spearmanr(yt, yp).correlation if len(yt) > 2 and np.std(yp) > 0 else float("nan")
        return {
            f"{prefix}mae": mae,
            f"{prefix}spearman": float(corr) if corr is not None else float("nan"),
            f"{prefix}n_scored": int(mask.sum()),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("regressor is not fitted")
        return self.model.predict(np.asarray(X, dtype=float))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "feature_names": self.feature_names,
                    "params": self.params,
                    "model": self.model,
                    "target_name": self.target_name,
                    "fit_report": self.fit_report,
                },
                fh,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RegionRegressor":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        return cls(
            feature_names=tuple(payload["feature_names"]),
            params=payload.get("params", {}),
            model=payload["model"],
            target_name=payload.get("target_name", "target"),
            fit_report=payload.get("fit_report", {}),
        )


def counting_targets(detected_counts: np.ndarray, true_counts: np.ndarray) -> np.ndarray:
    """The proposed acquisition target: absolute counting error per region."""
    return np.abs(np.asarray(detected_counts, float) - np.asarray(true_counts, float))


def rank_by_prediction(region_ids: list[str], predictions: np.ndarray) -> list[tuple[str, float]]:
    """Descending ranking with a deterministic tie-break on region id."""
    pairs = list(zip(region_ids, predictions.tolist()))
    return sorted(pairs, key=lambda t: (-t[1], t[0]))
