# Evaluation protocol

Every convention below is implemented in `midogcx/evaluation/` and covered by tests.

## Pipeline order (fixed for all methods)

1. strict `score > threshold` filter,
2. whole-specimen duplicate suppression by centre distance (25 px),
3. one-to-one matching within 7.5 µm,
4. TP / FP / FN, counts, count error, signed bias.

Order matters: suppressing before matching means duplicates become false positives rather
than consuming ground-truth cells.

## Coordinate and unit handling

Boxes are **xyxy** in image pixels. Matching distances are converted to **micrometres** using
each specimen's own spacing:

`d_um = sqrt((dx · sx_um)² + (dy · sy_um)²)`

Spacing is read from the TIFF resolution tags; the source (`tiff_resolution`,
`imagej_metadata` or `default_assumption`) is recorded per specimen in
`data/splits/geometry.csv`. Anisotropic spacing is therefore handled, and a specimen that
falls back to 0.25 µm/px is visible rather than silently assumed.

The official evaluator expresses the same radius as `7.5e-3` in millimetre coordinates. At
0.25 µm/px the two agree; the physical formulation additionally survives anisotropic files.

## Matching

Greedy global-nearest, one-to-one:

1. enumerate every (prediction, ground-truth) pair within 7.5 µm,
2. sort by distance, then prediction index, then ground-truth index,
3. accept a pair when neither endpoint is already used.

Consequences: matching is deterministic; no ground-truth cell is used twice; no prediction is
matched twice. Boundary semantics: `distance ≤ radius` matches.

Reported per specimen: `mean_match_distance_um`, so a detector can be seen drifting toward the
radius limit even when F1 looks stable.

**Known gap.** Greedy assignment is not proven identical to `evalutils` 0.3.1 on crowded,
ambiguous point sets. The blueprint records agreement on eight synthetic cases (exact, just
inside/outside radius, duplicates, two targets, empty GT, empty predictions, both empty); all
eight are reproduced in `tests/test_matching.py`. Real-prediction parity with the official
scorer has **not** been established and is not claimed.

## Duplicate suppression

Greedy centre-distance suppression: visit predictions by descending score (ties by original
index), keep each, remove every remaining point within 25 px. Implemented with a KD-tree.

25 px is the official whole-image radius and is a **pixel** quantity, so it is not physically
constant across scanners. It is applied as published; the discrepancy with the 30 px
(7.5 µm) matching radius is a property of the official protocol, not of this code.

## Threshold selection

* Search range `0.50 … 0.99`, step `0.01` — the official range.
* Criterion: pooled F1 on the **source calibration** specimens.
* The selected threshold is then applied unchanged to every method and to target assessment.
  Recalibrating per method on target data would leak.
* The full sweep is saved (`threshold.json`) with its search range and criterion, so a
  reader can see it was not an official calibration and not a per-method optimum.

Earlier exploratory sweeps in the pilot used 0.10–0.50; those are not comparable and are not
reproduced here.

## Metric definitions

| Metric | Definition |
|---|---|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2TP / (2TP + FP + FN) |
| Predicted count | TP + FP (detections surviving threshold + NMS) |
| Ground-truth count | TP + FN |
| Absolute count error | \|N_pred − N_GT\| = \|FP − FN\| |
| Signed bias | N_pred − N_GT |
| MAE | mean over specimens of absolute count error |

## Empty-case conventions (predefined)

| Case | Precision | Recall | F1 |
|---|---|---|---|
| No predictions, no GT | 0.0 | 0.0 | **0.0** |
| No predictions, GT present | 0.0 | 0.0 | 0.0 |
| Predictions, no GT | 0.0 | 0.0 | 0.0 |

**Deliberate deviation.** The official evaluator adds `1e-6` to numerators and denominators,
which makes an all-empty specimen score **F1 = 1.0** — a detector that emits nothing would
be rewarded. This project reports 0.0 for a zero-denominator metric. Pooled aggregates over
many specimens are numerically unaffected because an empty specimen contributes no TP/FP/FN,
so comparisons against published pooled numbers remain meaningful.

## Aggregation

* **Pooled**: sum TP/FP/FN across specimens, then divide. Dominated by large specimens.
* **Macro**: mean of per-specimen metrics. Every specimen counts equally.
* Both are always reported; they can disagree sharply on imbalanced cohorts.
* Uncertainty: specimen-level bootstrap, 1000 resamples for single-method summaries, 2000 for
  paired comparisons, fixed seed recorded in the output.

## Paired comparison

`paired_comparison(A, B)` requires identical specimen lists in the same order and reports
`mean(A − B)` with a bootstrap CI for absolute count error, F1 and signed bias. Negative ΔMAE
favours A. Seeds are not treated as extra samples.

## What every result file contains

```
metrics/<method>.json     pooled, macro, counting, CIs, per-specimen rows,
                          threshold, NMS radius, match radius
metrics/<method>_predictions.json   raw points + scores (threshold-reappliable)
selections/<strategy>.json          full ranking, admitted regions, budget used
predictor_report.json               OOF and in-sample quality for both learned rules
paired_comparisons.json             per-method difference vs source_only
comparison.csv                      one row per method
```

## Verification status

| Item | Status |
|---|---|
| Matcher: 8 synthetic cases from the blueprint | reproduced in tests |
| Matcher: determinism and one-to-one property | tested |
| NMS: cluster, boundary, tie-break, strict threshold | tested |
| Metrics: cancellation case (45 FP / 42 FN → error 3) | tested |
| Empty-case conventions | tested |
| Threshold sweep on saved predictions | tested |
| Physical spacing from TIFF tags | exercised in the smoke test |
| Parity with `evalutils` on real predictions | **not established** |
| Official full-benchmark equivalence | **not claimed** |
