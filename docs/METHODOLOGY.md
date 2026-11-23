# Methodology

Implementation decisions, why they were made, and what would falsify the claim.

## 1. Design

Controlled retrospective experiment. One source domain (human breast carcinoma), one target
domain (human cutaneous melanoma), one detector family, one annotation-area budget per
comparison. Five acquisition rules are compared; only the rule differs.

**Primary question.** Under an equal annotated tissue-area budget, does ranking melanoma
regions by predicted mitotic counting error reduce counting error on independent melanoma
specimens more than the four baselines?

**Primary endpoint.** Specimen-level mean absolute counting error (MAE) on target
assessment specimens.

**Co-primary / required companion endpoint.** Localisation: precision, recall, F1, and the
raw FP/FN counts. A count-error win with collapsed localisation is not a win; FP and FN
cancel inside `|N_pred − N_GT|` and the blueprint documents a specimen (404) where 45 FP and
42 FN produced a count error of 3.

**No superiority is claimed in advance.** A well-supported negative result is a valid
outcome: if predicted count error merely duplicates density or loss, that is a finding.

## 2. Data separation

The 55 specimens in `data/splits/cohort_v1.json` split into five roles, enforced disjoint by
`midogcx/data/splits.py`:

| Role | n | Tumour | Purpose |
|---|---:|---|---|
| `source_train` | 20 | breast | fits the source detector only |
| `source_calib` | 5 | breast | threshold selection; also predictor supervision |
| `source_assess` | 8 | breast | source-domain assessment; also predictor supervision |
| `target_pool` | 12 | melanoma | **unlabelled** candidates for acquisition |
| `target_assess` | 10 | melanoma | independent target assessment |

Rules the code enforces:

* A specimen can appear in exactly one role (validation raises otherwise).
* Target assessment specimens never select a threshold, rank a region, tune anything, or
  contribute supervision.
* The 17 specimens used by the historical Mac pilot are excluded outright; they have been
  seen during development and cannot serve as pristine evidence.
* All 55 are official **training** records. The official test split is untouched.

## 3. Regions and the area budget

Candidates are a **non-overlapping** 512×512 grid of tissue-bearing regions per specimen
(tissue fraction ≥ 0.5 from an Otsu mask). Non-overlap is a deliberate design choice: it
makes "unique annotated tissue area" exactly additive and removes double counting from the
budget arithmetic.

Tissue area is **physical**: `tissue_pixels × spacing_x_um × spacing_y_um`, with spacing read
from each TIFF's resolution tags and the source recorded per specimen.

The budget is 20% of the pool's total tissue area (the protocol's feasibility fraction).
Regions are admitted in rank order; a region that would overshoot the remaining budget is
skipped and the walk continues. Every strategy therefore ends with the same total tissue
area (up to one region), not the same patch count.

**Equal area is not equal expert time.** Nothing in this project measures annotation effort.

## 4. Detector

`torchvision` RetinaNet, ResNet-50-FPN-v2 backbone, ImageNet/COCO initialisation, fine-tuned
for one foreground class.

The class convention is the single most error-prone detail in the previous work, so it is
resolved from the installed library source and pinned in one place:

* `num_classes = 2` **includes** background, so the mitotic-figure class is label **1**.
* MIDOG++ category 1 maps to label 1; category 2 (`not mitotic figure`) is a curated hard
  negative and is **never** a foreground box.
* Post-processing keeps `labels == 1`, matching the official evaluator's `cls == 1` filter.

COCO weights initialise the backbone, FPN and box head. The classification head cannot be
reused across class counts, so it is reinitialised — and that skip is asserted key-by-key
rather than accepted silently. COCO initialisation is not MIDOG++ pretraining and is never
described as such.

## 5. The five acquisition rules

All rules see only unlabelled information: detector outputs, region geometry and tissue mask.

1. **Random** — seeded uniform permutation.
2. **Uncertainty** — Σ h(s) over detections within ±0.1 of the threshold, h the binary
   entropy. A region with no borderline detection scores 0. All candidates have the same
   pixel size, so this aggregate is not length-biased; edge regions are slightly smaller and
   this is reported.
3. **Density** — predicted mitoses per mm² of tissue (thresholded count / area).
4. **Learned loss** — a gradient-boosted regressor predicting the detector's training loss on
   that region (the generic predicted-loss criterion).
5. **Count error (proposed)** — the same regressor family, same features, same protocol, but
   the target is `|n_r − y_r|`, the absolute counting error the region exhibits under the
   locked threshold.

Rules 4 and 5 differ **only in their target**. That is what makes the comparison a comparison
of criteria rather than of model capacity. Both are fitted with `GroupKFold` by specimen and
report out-of-fold and in-sample quality side by side.

An optional sixth rule (`scale_disagreement`, count difference between two inference scales)
is implemented but excluded from the primary five. It is motivated by Xu et al. (2026) and is
**not** a faithful reproduction of their method.

## 6. Leakage control

Three distinct leak paths, each closed by construction:

1. **Target labels never reach a selector.** Feature rows for pool regions are built from
   detector outputs and geometry; `attach_targets` is only ever called for the error-training
   cohort. Nothing in `selection/` accepts a ground-truth array.
2. **Predictor targets are out-of-sample.** Supervision comes from `source_calib` and
   `source_assess`, which never entered source-detector fitting. This is weaker than
   k-fold cross-fitted detectors (rejected as unaffordable within the timeline) and the
   limitation is recorded, not hidden.
3. **Assessment is independent.** Threshold selection, ranking, predictor fitting and all
   hyperparameters are fixed before any target assessment specimen is scored.

A residual risk remains and is stated openly: the predictor is trained on breast tissue and
applied to melanoma. Whether the error signal transfers *is* the research question, so a null
result could mean "the criterion is no better" or "the predictor did not transfer". The
out-of-fold Spearman correlation on the source domain does not settle this; the per-region
rank quality on target data that the run reports does.

## 7. Fairness controls between branches

Every branch shares: the same source checkpoint (reloaded fresh, not continued), the same
candidate pool, the same budget fraction, the same number of updates, optimiser, schedule,
augmentation, sampling ratio, seed, locked threshold, NMS radius and evaluation code. Only
the selected regions differ.

Without these controls a difference between branches could come from any of them.

## 8. Statistics

* Specimens are the unit of analysis. Patches, tiles and repeated detections are not
  independent observations.
* Aggregates are reported **pooled** (sum TP/FP/FN then divide) and **macro** (mean over
  specimens), because they answer different questions on imbalanced specimens.
* Uncertainty: specimen-level bootstrap (1000–2000 resamples, fixed seed).
* Strategy comparisons are **paired** by specimen, reporting mean differences with bootstrap
  CIs. Seed variability is reported separately and never pooled with specimen variability.
* Predefine the primary comparison (proposed vs random) and the primary budget (20%) before
  looking at target assessment results, to avoid cherry-picking.
* Two seeds support a feasibility judgement, not a significance claim.

## 9. What would falsify the claim

* Count-error selection is no better than random at equal area, in both seeds, with
  overlapping CIs.
* Count-error selection is indistinguishable from density or learned-loss selection
  (top-20 overlap near 1.0, near-identical MAE).
* The proposed rule wins on MAE only by producing fewer detections, i.e. precision/recall and
  F1 degrade beyond what a counting improvement justifies.
* No adaptation branch improves on the unadapted `source_only` control, which would mean the
  adaptation setting itself does not work and no acquisition comparison is interpretable.

## 10. Threats to validity

1. **Confounded domain shift.** Melanoma is single-origin (UMC Utrecht), single-scanner
   (Hamamatsu XR). Tumour type and acquisition cannot be separated. No multi-site claim.
2. **Development data.** Official training records used throughout; the official test split is
   untouched and reserved for a final locked evaluation.
3. **Patient independence unproven.** The dataset publishes no patient identifiers.
4. **Small cohorts.** 12 pool and 10 assessment specimens. Wide intervals are expected.
5. **Annotation area ≠ effort.** Budget equality is measured in mm², not pathologist minutes.
6. **Single detector family.** Conclusions are about RetinaNet-R50-FPN, not detectors in
   general.
7. **Predictor transfer unknown** (see §6).
