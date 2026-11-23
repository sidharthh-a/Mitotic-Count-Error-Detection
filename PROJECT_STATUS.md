# Project Status

Last updated: 9 November 2026. Update this file at the end of every working session.

---

## Completed

### Phase 1 — research documents and environment audit
- Read the portable blueprint and the literature presentation in full.
- Audited the actual machine (not the historical Mac): Windows 11, i5-1155G7, 7.7 GiB RAM,
  Intel Iris Xe (no CUDA/MPS), Python 3.13.15, torch 2.7.0+cpu, `C:` 158 GB free,
  `D:` (project) 3.9 GB free. → `docs/ENVIRONMENT_AUDIT.md`
- Recommended and adopted: **train on Colab, develop/audit/evaluate locally.**

### Phase 2 — dataset audit (no images downloaded)
- Downloaded and verified the official manifest and annotation JSON.
- Built the full Figshare file/size inventory (506 files, **65.05 GiB**).
- Re-derived every published count from the files: 503 eligible specimens, 553 JSON image
  records, 50 JSON-only records (151–200) with **zero** annotations, 26,286 annotations,
  11,937 positive / 14,349 non-mitotic, 254 human / 249 canine.
- Verified the **xyxy** box convention and the category mapping.
- Verified the scanner alias `Hamammatsu XR` → `Hamamatsu XR` with explicit evidence.
- Measured the cohort download exactly: **55 specimens, 7.52 GiB**.
- → `docs/DATASET_AUDIT.md`, `data/metadata/audit.json`

### Phase 3 — cohort freeze
- Transcribed the blueprint's 55-specimen cohort and validated it against the manifest:
  all present, correct tumour, official-train only, disjoint roles, no historical pilot
  reuse. → `data/splits/cohort_v1.json` (digest `521a7466…`)
- Recorded domain confounding: melanoma is single-origin, single-scanner.

### Phase 4 — implementation
- Complete package `midogcx/` (data, models, selection, inference, adaptation, evaluation).
- RetinaNet label convention resolved from the installed torchvision source and verified:
  `num_classes=2` includes background, foreground label is **1**, only
  `cls_logits` is reinitialised from COCO and that is asserted key-by-key.
- Five acquisition strategies defined precisely; optional sixth (`scale_disagreement`) coded
  but excluded from the primary comparison.
- Equal-tissue-area budget enforcement with overshoot skipping.
- Out-of-fold error-predictor fitting by specimen, reporting OOF and in-sample quality
  side by side.
- Six scripts: inventory audit, dataset audit, cohort build, download, region prep, train,
  infer, experiment, smoke test.
- Streamlit interface with four tabs, driven only by artefacts that exist on disk.
- **98 automated tests passing.**

### Phase 5 — verification performed
- `python -m pytest tests/ -q` → **98 passed** (now 108, see Phase 6).
- `python scripts/smoke_test.py` → passes; exercises the whole pipeline on synthetic TIFFs.
- `python scripts/smoke_test.py --pretrained coco` → passes; confirms only the
  classification head is reinitialised.
- Streamlit app exercised through `streamlit.testing.v1.AppTest` → **no exceptions**.
- Measured CPU throughput: **2.4–2.8 s per update** (batch 1, 512×512, R50-FPN).

### Phase 6 — data acquired, then the final readiness gate

**Data now on disk.** 55/55 cohort specimens downloaded to `C:/midogpp_data` (7.6 GiB),
byte-verified; the 55th file needed one resume after a truncated transfer. Region
preparation then produced **2,395 regions** across 55 specimens:

| Role | Regions | Tissue (mm²) | Positives |
|---|---:|---:|---:|
| source_train | 745 | 5.910 | 25 |
| source_calib | 232 | 1.870 | 7 |
| source_assess | 455 | 3.766 | 19 |
| target_pool | 468 | 3.620 | 38 |
| target_assess | 495 | 4.054 | 44 |

Spacing resolved from TIFF tags for all 55 specimens. Normalisation mean
`[0.7953, 0.5890, 0.7460]`, std `[0.1145, 0.1550, 0.1049]`, computed from `source_train` only.

**Readiness gate: 17/17 checks passed** → full detail in `EXPERIMENTAL_READINESS_AUDIT.md`.

| Check | Measured result |
|---|---|
| pytest | **108 passed** (51.7 s); **115** after the baseline-gate fix below |
| synthetic end-to-end smoke test | **PASSED** (127.9 s) |
| frozen cohort | digest `521a7466…` matches file, roles yaml and expected composition |
| frozen region grid | 2,395 regions recomputed identically, nothing written |
| BatchNorm buffers | 159 buffers unchanged across sliding-window inference, region inference and per-region loss; train mode restored |
| five strategies | same 468-region pool; budget 0.724 mm² each; spread 0.000253 mm² vs one region 0.013561 mm² |
| checkpoint save/reload | 362 tensors identical, 170 trainable identical, provenance intact, no pretrained fetch |
| Colab notebook | 24 cells valid; frozen artefacts verified via `--check`, never regenerated |
| frozen artefacts | 5 files byte-identical before/after the pass |

**Two defects found and fixed by this gate:**
1. `load_cohort` never verified its own `payload_sha256`, so a modified split would have
   loaded silently. It now raises on mismatch (test added).
2. The Colab notebook both regenerated the frozen region grid and contained a literal `\n`
   in the baseline-gate command. `prepare_regions.py --check` was added; the notebook now
   verifies instead of rewriting, and the broken command was replaced.

**Throughput note.** Run-to-run CPU variance is about 2× (4.15 s/update earlier vs
7.72 s/update in this pass; whole-slide inference 19.5 s vs 39.4 s). Plan CPU budgets from
the slow end.

---

## Pending

| # | Task | Blocked by |
|---|---|---|
| 1 | ~~Download the 55-specimen cohort~~ | **done** — 55/55 at `C:/midogpp_data` |
| 2 | ~~`prepare_regions.py` on real data~~ | **done** — 2,395 regions, grid verified |
| 3 | ~~Source detector training to a usable checkpoint~~ | **done** — 500 updates on Colab, checkpoint verified (see Model training status) |
| 4 | Threshold calibration + `source_only` control evaluated | Task 3 |
| 5 | Error-predictor fit on real regions (OOF report) | Tasks 3–4 |
| 6 | Five-strategy adaptation comparison | Tasks 3–5 |
| 7 | Second seed; paired/uncertainty analysis | Task 6 |
| 8 | Real figures and result tables in the interface | Task 6 |

---

## Current blockers

**Blocker 1 — resolved.** The cohort lives on `C:/midogpp_data` (7.6 GiB), outside the
3.9 GB project drive, and is selected automatically through `MIDOG_DATA_ROOT`.

**Blocker 2 — resolved.** Download approved and completed; 55/55 files byte-verified.

**Blocker 3 — GPU for real training (only remaining blocker).** No CUDA device locally.
Measured CPU throughput (4.2–7.7 s/update at batch 1) makes a 500-update branch ≈ 35–65 min
and a full 6-branch run with 60 whole-slide inferences ≈ 10–16 hours of continuous laptop
CPU. Technically possible overnight, thermally risky, and far slower than Colab.

`notebooks/colab_pipeline.ipynb` is ready and passes the static checks; the source baseline
is the next action. Nothing else is waiting on the local machine.

---

## Dataset status

| | |
|---|---|
| Metadata | downloaded, hashed, audited |
| Images on disk | **55/55 specimens**, 7.6 GiB at `C:/midogpp_data` |
| Region grid | frozen: 2,395 regions, verified reproducible |
| Cohort | frozen and validated (55 specimens, digest `521a7466…`) |
| Official test split | deliberately untouched |

## Model training status

One real run exists: the source-domain baseline, trained on Colab on 23 November 2026 and
handed back as `outputs/runs/source_baseline/checkpoint.pth` (145.7 MB).

| | |
|---|---|
| Updates | 500 in 382.4 s (0.76 s/update) |
| Role / regions | `source_train`, 745 regions, 20 specimens |
| Architecture | `retinanet_resnet50_fpn_v2`, 362 state tensors, 36.4 M params, 170 trainable |
| Config fingerprint | `778e4f9a6970900c…` — reproduces exactly from the shipped `base.yaml` |
| Effective settings | defaults, i.e. `batch_size=2`, 500 updates; **not** the runbook's `batch_size=4` |
| Provenance | normalisation identical to frozen `outputs/metadata/normalization.json`; detector config identical to `base.yaml`; strict `load_state_dict` passes with the head shape checked |
| Integrity | no NaN/Inf in any tensor; BatchNorm buffers are trained values, not initial ones |

Learning probe (not a metric): total detection loss on 3 held-out `source_assess` regions
(2 mitotic, 1 empty) is **0.456** for the trained weights versus **1.728** for a
random-initialised model of the same architecture — the weights carry real signal. No
precision/recall/MAE has been measured yet, so **no performance claim is made**.

Everything reported about training before this run is either synthetic smoke-test execution
or a CPU throughput measurement.

## Experiment status

No acquisition experiment has been run on real specimens. `run_acquisition_experiment.py`
is implemented, and its selection/budget arithmetic is verified against the frozen grid with
synthetic detections, but no real predictions exist.

## Baseline-gate fix (9 November 2026)

The first Cell 16 run crashed before evaluating anything:

```
TypeError: HistGradientBoostingRegressor.__init__() got an unexpected keyword argument 'n_splits'
```

`configs/base.yaml` keeps `n_splits` inside `acquisition.error_predictor`, and that whole
block was forwarded as estimator hyperparameters, so `n_splits` reached sklearn. Fixed at
two layers: `split_cv_params()` separates the CV key from the hyperparameters (used by
`fit_predictors` for **both** predictors), and `RegionRegressor._new_model()` strips CV keys
as defence in depth while still rejecting genuinely unknown parameters instead of ignoring
them. The effective fold count is now recorded (`n_splits`, `n_groups`) because `GroupKFold`
caps the fold count at the number of specimens.

Seven regression tests were added in `tests/test_error_predictor.py`, including one that
fits with the **shipped** config block, which is the case that actually failed.

## Readiness gate

`python scripts/verify_readiness.py` → **17/17 passed**, frozen artefacts byte-identical
afterwards. The gate clears compute allocation; it produces no scientific result.
See `EXPERIMENTAL_READINESS_AUDIT.md`.

---

## Important decisions

1. **Package, not flat folders.** All logic in `midogcx/`, so tests import the real code and
   Colab installs one directory. The brief allowed restructuring.
2. **Non-overlapping 512×512 region grid.** Makes "unique annotated tissue area" exactly
   additive, removing an entire class of double-counting error from the budget comparison.
3. **Error-predictor supervision from `source_calib` + `source_assess`.** These specimens
   never entered source-detector fitting, so predictor features are out-of-sample. Training
   k cross-fitted detectors was rejected as unaffordable within the timeline; the trade-off
   is recorded rather than hidden.
4. **Empty-case metrics return 0.0, not the official epsilon value.** The official formula
   scores an all-empty specimen F1 = 1.0, which would reward a detector for emitting
   nothing. Pooled aggregates are numerically unaffected.
5. **Greedy global-nearest matching.** Deterministic and one-to-one; not proven identical to
   `evalutils` on crowded ambiguous cases. Documented in `docs/EVALUATION_PROTOCOL.md`.
6. **Low inference score floor (0.01), thresholds applied offline.** Lets threshold
   selection and re-evaluation run without repeating inference.
7. **No `strict=False` checkpoint loading.** Shape-mismatched tensors are filtered and the
   skip list is asserted, so an uninitialised head cannot masquerade as pretrained.
8. **Frozen artefacts are self-verifying.** `load_cohort` recomputes `payload_sha256` and
   refuses a modified split; `prepare_regions.py --check` recomputes the grid and refuses a
   non-reproducible one. A remote runtime can verify but never rewrite either.
9. **Verification is a script, not a checklist.** `scripts/verify_readiness.py` re-runs all
   seven gates and exits non-zero on failure, so the audit is reproducible rather than a
   claim in a document.
10. **Cross-validation settings never reach an estimator.** `n_splits` configures
    `GroupKFold` and is stripped from the predictor parameter block before any sklearn
    constructor sees it, so a config edit cannot crash the baseline gate.
11. **The adaptation trainer's log wording is generic, not a data claim.** Its
    `N target + M source regions` line names argument slots, not cohort roles. The
    authoritative record is `meta.role` in the checkpoint and `n_target_regions` in
    `train_summary.json`.

---

## Exact commands to resume

```bash
cd D:/projects/Mitotic-Count-Error-Detection

# 0. sanity: code and tests (no data needed)
python -m pytest tests/ -q
python scripts/smoke_test.py

# 1. data is already in place; verify it rather than re-derive it
export MIDOG_DATA_ROOT="C:/midogpp_data"
python scripts/build_cohort.py --check
python scripts/prepare_regions.py --check

# 2. full readiness gate (17 checks, ~10 min; add --skip-regions to skip the grid recheck)
python scripts/verify_readiness.py

# 3. source detector (on Colab; see notebooks/colab_pipeline.ipynb)
python scripts/train_detector.py --tag source_baseline --set training.device=cuda

# 4. acquisition experiment
python scripts/run_acquisition_experiment.py \
    --checkpoint outputs/runs/<run>/checkpoint.pth --tag pilot --device cuda

# 5. interface
streamlit run frontend/app.py
```
