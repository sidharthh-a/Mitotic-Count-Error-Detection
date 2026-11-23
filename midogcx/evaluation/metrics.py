"""Detection and counting metrics, aggregation and paired comparison.

The protocol requires counting error to be reported *with* localisation (precision,
recall, F1, FP, FN), because counting error alone can be small while false positives and
false negatives cancel (the documented specimen-404 example: 45 FP and 42 FN give a count
error of 3 with weak localisation).

Conventions:
* ``precision = TP / (TP + FP)``, ``recall = TP / (TP + FN)``, ``F1 = 2TP / (2TP+FP+FN)``.
* **Empty-case convention (predefined, deliberately not the official epsilon trick):**
  a metric whose denominator is zero is reported as ``0.0``. The official MIDOG++
  evaluator adds ``1e-6`` to numerators and denominators, which makes an all-empty
  specimen score ``F1 = 1.0``; that would reward a detector for emitting nothing. Pooled
  aggregates over many specimens are numerically unaffected (an empty specimen adds no
  TP/FP/FN), so published comparisons stay meaningful.
* Pooled metrics sum TP/FP/FN across specimens. Macro metrics average per specimen.
* Counting: ``abs error = |N_pred - N_GT| = |FP - FN|``; ``signed bias = N_pred - N_GT``.
  MAE is the mean absolute specimen error. Specimens are the resampling unit; patches and
  tiles are never treated as independent observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

EPS = 1e-6


@dataclass
class SpecimenMetrics:
    slide_id: int
    n_pred: int
    n_gt: int
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_match_distance_um: float | None = None

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.true_positives + self.false_positives + self.false_negatives
        return (2 * self.true_positives) / denom if denom else 0.0

    @property
    def abs_count_error(self) -> int:
        return abs(self.n_pred - self.n_gt)

    @property
    def signed_bias(self) -> int:
        return self.n_pred - self.n_gt

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            abs_count_error=self.abs_count_error,
            signed_bias=self.signed_bias,
        )
        return out


def pooled_counts(results: list[SpecimenMetrics]) -> dict:
    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "n_predictions": sum(r.n_pred for r in results),
        "n_ground_truth": sum(r.n_gt for r in results),
    }


def macro_average(results: list[SpecimenMetrics]) -> dict:
    if not results:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(np.mean([r.precision for r in results])),
        "recall": float(np.mean([r.recall for r in results])),
        "f1": float(np.mean([r.f1 for r in results])),
    }


def counting_summary(results: list[SpecimenMetrics]) -> dict:
    if not results:
        return {"mae": 0.0, "mean_signed_bias": 0.0, "median_abs_error": 0.0}
    abs_err = np.array([r.abs_count_error for r in results], dtype=float)
    bias = np.array([r.signed_bias for r in results], dtype=float)
    return {
        "mae": float(abs_err.mean()),
        "median_abs_error": float(np.median(abs_err)),
        "mean_signed_bias": float(bias.mean()),
        "mean_predicted_count": float(np.mean([r.n_pred for r in results])),
        "mean_ground_truth_count": float(np.mean([r.n_gt for r in results])),
    }


def _bootstrap_ci(
    values: np.ndarray, statistic, n_boot: int, seed: int, alpha: float = 0.05
) -> list[float]:
    if n_boot <= 0 or len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic(values[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def aggregate_metrics(
    results: list[SpecimenMetrics],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Pooled + macro detection metrics, counting error, and specimen-bootstrap CIs."""
    if not results:
        raise ValueError("no specimen results to aggregate")
    abs_err = np.array([r.abs_count_error for r in results], dtype=float)
    bias = np.array([r.signed_bias for r in results], dtype=float)

    summary = {
        "n_specimens": len(results),
        "pooled": pooled_counts(results),
        "macro": macro_average(results),
        "counting": counting_summary(results),
        "counting_ci95": {
            "mae": _bootstrap_ci(abs_err, np.mean, n_boot, seed),
            "mean_signed_bias": _bootstrap_ci(bias, np.mean, n_boot, seed + 1),
        },
        "per_specimen": [r.to_dict() for r in results],
    }
    return summary


def paired_comparison(
    method_a: list[SpecimenMetrics],
    method_b: list[SpecimenMetrics],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired per-specimen difference (A - B) with a specimen bootstrap CI.

    Both inputs must contain the same specimens in the same order; pairing by specimen is
    what keeps the comparison honest when specimens differ in difficulty.
    """
    ids_a = [r.slide_id for r in method_a]
    ids_b = [r.slide_id for r in method_b]
    if ids_a != ids_b:
        raise ValueError("paired comparison requires identical specimen lists in the same order")
    if not ids_a:
        raise ValueError("no specimens to compare")

    d_mae = np.array([a.abs_count_error - b.abs_count_error for a, b in zip(method_a, method_b)], float)
    d_f1 = np.array([a.f1 - b.f1 for a, b in zip(method_a, method_b)], float)
    d_bias = np.array([a.signed_bias - b.signed_bias for a, b in zip(method_a, method_b)], float)

    def _mean(x: np.ndarray) -> float:
        return float(x.mean())

    return {
        "n_specimens": len(ids_a),
        "delta_abs_count_error": {
            "mean": _mean(d_mae),
            "ci95": _bootstrap_ci(d_mae, _mean, n_boot, seed),
        },
        "delta_f1": {"mean": _mean(d_f1), "ci95": _bootstrap_ci(d_f1, _mean, n_boot, seed + 1)},
        "delta_signed_bias": {
            "mean": _mean(d_bias),
            "ci95": _bootstrap_ci(d_bias, _mean, n_boot, seed + 2),
        },
        "per_specimen": [
            {
                "slide_id": sid,
                "delta_abs_count_error": float(dm),
                "delta_f1": float(df),
                "delta_signed_bias": float(db),
            }
            for sid, dm, df, db in zip(ids_a, d_mae, d_f1, d_bias)
        ],
    }


def metrics_table(per_run: dict[str, dict]) -> "object":
    """Build a comparison DataFrame from ``{method: aggregate_metrics(...)}`` output."""
    import pandas as pd

    rows = []
    for method, agg in per_run.items():
        pooled = agg["pooled"]
        counting = agg["counting"]
        rows.append(
            {
                "method": method,
                "n_specimens": agg["n_specimens"],
                "precision": pooled["precision"],
                "recall": pooled["recall"],
                "f1": pooled["f1"],
                "tp": pooled["tp"],
                "fp": pooled["fp"],
                "fn": pooled["fn"],
                "mae": counting["mae"],
                "mean_signed_bias": counting["mean_signed_bias"],
                "mae_ci_lo": agg["counting_ci95"]["mae"][0],
                "mae_ci_hi": agg["counting_ci95"]["mae"][1],
            }
        )
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
