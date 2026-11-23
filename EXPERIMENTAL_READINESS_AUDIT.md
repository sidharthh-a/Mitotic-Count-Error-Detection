# Experimental readiness audit

Gate that must be passed before any GPU time is spent. Every number below was measured on
this machine in the run recorded here; nothing is estimated.

- Date: 9 November 2026
- Commit state: working tree (no git repository yet)
- Host: Windows 11, i5-1155G7 (4C/8T), 7.7 GiB RAM, Intel Iris Xe, **no CUDA**
- Python 3.13.15, torch 2.7.0+cpu, torchvision 0.22.0+cpu
- Data root: `MIDOG_DATA_ROOT=C:/midogpp_data` (55 specimens, 7.6 GiB)
- Reproduce: `python scripts/verify_readiness.py` (add `--skip-regions` without images)

---

## Verdict

**17 of 17 checks passed. Cleared for source-baseline training on Colab.**

This clears the *machinery*. It does not produce, and must not be reported as, any
scientific result: no detector has been trained on MIDOG++ data yet.

---

## Frozen artefacts under observation

| Artefact | Role |
|---|---|
| `data/splits/cohort_v1.json` | immutable specimen split |
| `data/splits/cohort_v1_summary.json` | role composition + confounding report |
| `data/splits/regions.csv` | frozen 512×512 region grid (2,395 regions) |
| `data/splits/geometry.csv` | per-specimen physical spacing |
| `outputs/metadata/normalization.json` | channel statistics from source-train only |

All five were hashed before and after the verification pass and are **byte-identical**.

---

## 1. Frozen cohort unchanged — PASS

- Payload digest: `521a74669cc4462378344d883e9ed4d67f49a146b28d39fed388aed1c5374448`
- Recomputed from the file's roles: identical.
- Recomputed from the human-authored `configs/cohort_v1.yaml`: identical.
- Composition: `source_train` 20, `source_calib` 5, `source_assess` 8, `target_pool` 12,
  `target_assess` 10 = 55 specimens.

**Gap found and fixed during this audit.** `load_cohort` did not verify the recorded
`payload_sha256`, so a modified split would have loaded silently. It now raises on mismatch,
and `tests/test_splits.py::test_tampered_cohort_is_rejected` covers it.

## 2. Frozen region grid reproducible — PASS

`scripts/prepare_regions.py --check` recomputed the full grid from the 55 TIFFs and compared
it to the files on disk column by column, then wrote nothing:

```
frozen region grid verified (2395 regions, 55 specimens, normalisation matches); nothing written
```

Spacing came from TIFF resolution tags for all 55 specimens (`tiff_resolution`), so no
specimen falls back to an assumed 0.25 µm/px.

Checks compare `slide_id, x, y, width, height, tissue_px, n_positives, role`, physical
areas within 1e-6 µm², geometry within 1e-9, and normalisation within 1e-9.

Wall clock: ~7 min (reads every specimen once, ~117 Mpx each).

## 3. BatchNorm buffers unchanged — PASS

159 buffers snapshotted before and compared by exact tensor equality after each path:

| Path | Result |
|---|---|
| Sliding-window inference (`predict_array`, 1024×768) | 159 buffers unchanged, still `eval()` |
| Per-region inference (`predict_regions`, 4 regions) | unchanged, still `eval()` |
| Per-region loss (`detector_loss_per_region`, 2 regions) | unchanged, `train()` restored |

The third path is the dangerous one: torchvision detection losses only exist in `train()`
mode, and entering it would otherwise overwrite every running statistic with single-image
statistics from source crops — silently changing the detector before it produces the pool
features and the `source_only` control. The function snapshots and restores all buffers and
the previous mode in a `finally` block.

Also covered by `tests/test_readiness.py::test_region_loss_computation_preserves_buffers_and_mode`.

**Recorded consequence:** the learned-loss target is computed under batch statistics rather
than running estimates, so it is a slightly different normalisation regime from inference.
Inherent to torchvision's loss API; it affects only the scalar loss target, identically for
both learned rules.

## 4. Five strategies aligned, budgets identical — PASS

Run on the **real** frozen grid: 468 `target_pool` regions, verified non-overlapping.

Synthetic detections (seeded) and a stub predictor are used deliberately: this check tests
budget and alignment arithmetic, not ranking quality.

| Strategy | Same pool | Regions selected | Tissue (mm²) | Utilisation |
|---|---:|:---:|---:|---:|
| random | yes | 99 | 0.723900 | 100.0% |
| uncertainty | yes | 102 | 0.723900 | 100.0% |
| density | yes | 144 | 0.723700 | 100.0% |
| learned_loss | yes | 109 | 0.724000 | 100.0% |
| count_error | yes | 109 | 0.724000 | 100.0% |

- Budget = 20% of pool tissue = **0.724 mm²**.
- Spread across strategies: **0.000253 mm²**, far inside one region (0.013561 mm²).
- Every strategy ranked all 468 regions exactly once; no region admitted twice.

Region counts differ (99 vs 144) while area is equal — that is the intended behaviour:
equal annotated **area**, not equal patch count. A high-density strategy prefers small
high-density regions and needs more of them to spend the same area.

**Caveat:** `learned_loss` and `count_error` share one stub predictor in this check, so their
identical counts are expected and say nothing about the real rules. Their independence is
exercised in `tests/test_readiness.py::test_all_strategies_share_pool_and_budget` and will be
measured for real on the first genuine experiment (`ranking_overlap.json`).

## 5. Checkpoint save/reload — PASS

- 362 state tensors saved and reloaded: **all exactly equal** (after deterministic weight
  perturbation, so a no-op reload could not pass).
- 170 trainable tensors: identical set after reload, including the backbone-freezing pattern
  (`layer3`/`layer4`/head trainable, `layer1` not).
- Provenance survives: detector config, normalisation mean/std, `meta.updates=123`, seed.
- Reload builds the architecture with `pretrained="none"` and overwrites it, so no
  ImageNet/COCO download happens on a fresh runtime (`reinitialised == ["all"]`), and the
  model returns in `eval()` mode.
- Head-size guard rejects a checkpoint whose `cls_logits` does not match
  `num_classes × 9 anchors` (covered in `tests/test_detector.py`).

## 6. Colab notebook — PASS (after two fixes)

Structure: 24 cells, valid notebook JSON, every code cell free of stray backslashes.

Two real defects were found and fixed in this pass:

1. **Cell 16 contained a literal `\n`** instead of a line continuation, so the baseline-gate
   command would have failed on Colab. Replaced with a single-line command.
2. **Cell 11 regenerated the frozen region grid** (`prepare_regions.py`), which a fresh
   runtime could silently rewrite under a different library version. `--check` mode was
   added and the notebook now calls it.

Both frozen artefacts are therefore only ever **verified**, never regenerated:

```
build_cohort.py --check       -> cohort verified unchanged, nothing written
prepare_regions.py --check    -> grid verified, nothing written
```

The notebook also refuses to train unless `torch.cuda.is_available()`.

**Not verified:** an actual Colab session. Colab is not reachable from this machine, so
"runs on a fresh CUDA runtime" is a static and local-substitute claim, not an observed one.
The first Colab run must therefore start with the test cell and the two `--check` cells and
only proceed if they pass.

## 7. Test suite and smoke test — PASS

| Check | Result |
|---|---|
| `python -m pytest tests/ -q` | **108 passed**, 5 warnings, 51.7 s — then **115 passed** after the amendment below |
| `python scripts/smoke_test.py` | **PASSED** in 127.9 s (synthetic TIFFs) |
| `python scripts/verify_readiness.py` | **17/17 checks passed** |

Smoke test on final code: metadata join, geometry, region grid, cohort validation, detector
construction, 2 training updates, whole-slide inference, region inference, threshold sweep,
matching/evaluation, three ranking strategies, per-region loss.

**Coverage gap, stated explicitly:** the smoke test skips `learned_loss` and `count_error`
because the synthetic set has fewer than two groups for out-of-fold fitting. Those two rules
are covered by the readiness test with stub predictors, not end-to-end on real data.

---

## Amendment — baseline-gate crash (9 November 2026)

The gate above passed on machinery, but the first real Cell 16 run failed before producing
any prediction:

```
TypeError: HistGradientBoostingRegressor.__init__() got an unexpected keyword argument 'n_splits'
```

**Cause.** `configs/base.yaml` nests `n_splits: 5` inside `acquisition.error_predictor`,
beside the boosting hyperparameters. `run_acquisition_experiment.py` passed that whole block
as `params`, and `RegionRegressor._new_model()` forwarded it to the estimator, which has no
such keyword. `n_splits` is a `GroupKFold` fold count — an argument of `fit()`.

**Why the gate did not catch it.** `verify_readiness.py` and the smoke test both exercised
the two learned strategies with **stub predictors**, never constructing
`RegionRegressor` from the shipped config block. The strategy/budget invariants were real;
the estimator construction path was untested. That is the gap this amendment closes.

**Fix.** `split_cv_params()` separates the CV key from the hyperparameters and is applied in
`fit_predictors` for both the count-error and the detector-loss model; `_new_model()` strips
CV keys defensively and raises on any other unknown parameter rather than ignoring it. The
report now records the effective fold count and group count, separately from the requested
value.

**Result.** `tests/test_error_predictor.py` (7 tests) fits both predictors with the shipped
config block, asserts grouped folds never split a specimen, asserts every region is held out
exactly once, and pins the error for unknown parameters. Full suite: **115 passed**.
No frozen artefact, dataset file or checkpoint was touched.

## Throughput measured during this pass

| Run | Observed |
|---|---|
| Earlier smoke test | 4.15 s/update (batch 1, 512², R50-FPN, CPU) |
| This pass, smoke test | 7.72 s/update |
| This pass, whole-slide inference (2048×1536) | 39.4 s (earlier: 19.5 s) |

Run-to-run variance on this laptop is roughly **2×**, consistent with a 4-core mobile CPU
thermally throttling and sharing the machine. Any CPU plan must budget from the slow end.

---

## Residual risks accepted before training

1. **No real detector yet.** Every number above concerns machinery. Detection quality is
   unknown until the source baseline trains.
2. **Baseline gate not yet exercised on real data.** `--stop-after-control` exists so the
   unadapted control is inspected before any branch is trained; use it.
3. **Bootstrap/paired statistics untested at production scale** (1000–2000 resamples over
   10 assessment specimens). Logic is unit-tested; runtime cost is not.
4. **Matcher parity with `evalutils` remains unproven** on real crowded predictions
   (documented in `docs/EVALUATION_PROTOCOL.md`).
5. **Target domain is single-origin, single-scanner.** Tumour and acquisition effects cannot
   be separated; no multi-site claim is permitted.

---

## Go/no-go

| Gate | Status |
|---|---|
| Frozen cohort immutable and self-verifying | GO |
| Frozen region grid reproducible | GO |
| Inference/loss paths side-effect free | GO |
| Strategies aligned at equal physical budget | GO |
| Checkpoints faithful and self-describing | GO |
| Notebook safe on a fresh runtime | GO (static + local substitute) |
| GPU training | **Proceed to Colab source baseline** |
