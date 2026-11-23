# Count-Error-Guided Region Selection for Cross-Domain Mitotic Figure Detection

**Formal title:** Evaluating Count-Targeted Region Selection for Label-Efficient Cross-Domain
Mitotic Figure Detection: A MIDOG++ Study.

**Research question.** Under equal annotated tissue-area budgets, does selecting melanoma
tissue regions by *predicted mitotic-counting error* improve a breast-trained detector's
adaptation, compared with random selection, detector uncertainty, predicted mitotic density
and learned detector loss?

This repository is a working, tested implementation of that study. It is a research
prototype on retrospective MIDOG++ specimen images — **not** a diagnostic tool, and it has
**no** clinical validation.

---

## Status: what is real right now

Read this before quoting anything.

| Item | State |
|---|---|
| Dataset metadata audit (503 specimens, 11,937 positives) | **Done**, verified from the official files |
| Storage audit (65.05 GiB collection; 7.52 GiB cohort) | **Done**, exact per-specimen sizes |
| 55-specimen cohort frozen and validated | **Done**, `data/splits/cohort_v1.json` |
| Full pipeline code (data → detector → selection → adaptation → evaluation) | **Done** |
| Automated tests | **108 passing** |
| End-to-end smoke test on synthetic TIFFs | **Passes**; 4.2–7.7 s per training update on this CPU |
| Specimen images downloaded | **55/55**, 7.6 GiB at `C:/midogpp_data` (outside the project drive) |
| Frozen region grid | **2,395 regions**, verified reproducible |
| Pre-training readiness gate | **17/17 checks passed** — `EXPERIMENTAL_READINESS_AUDIT.md` |
| Source detector trained on real MIDOG++ data | **No** — needs GPU (see `docs/COMPUTE.md`) |
| Acquisition experiment results | **No** — nothing has been run on real data |
| Streamlit interface | **Runs**, shows the audit; detections need images + a checkpoint |

There are no experimental results in this repository yet, and none are invented anywhere in
the code, docs or interface. The blueprint's earlier Mac pilot numbers are historical
user-reported values; they are not reproduced here and must not be pooled with anything this
code produces.

---

## Environment (measured, not assumed)

| | |
|---|---|
| OS / CPU / RAM | Windows 11 · i5-1155G7 (4C/8T) · 7.7 GiB |
| GPU | Intel Iris Xe — no CUDA, no MPS |
| Python / torch | 3.13.15 · 2.7.0+cpu · torchvision 0.22.0+cpu |
| Disk | `C:` 158 GB free · `D:` (project) **3.9 GB free** |

Two consequences drive everything:

1. **Train on a GPU (Colab).** CPU training is possible but slow; measured throughput is
   ~2.5 s per update at batch size 1, 512×512, ResNet-50-FPN.
2. **The project drive cannot hold the images.** The 55-specimen cohort is 7.52 GiB. Images
   therefore live outside the repo and are located through `MIDOG_DATA_ROOT`.

Full detail: `docs/ENVIRONMENT_AUDIT.md`.

---

## Quick start (no dataset needed)

```bash
python -m pip install -r requirements.txt

python scripts/audit_dataset_inventory.py   # Figshare file/size inventory (cached)
python scripts/audit_dataset.py             # -> docs/DATASET_AUDIT.md + data/metadata/audit.json
python scripts/build_cohort.py              # -> data/splits/cohort_v1.json (validated)

python -m pytest tests/ -q                  # 108 tests
python scripts/smoke_test.py                # full pipeline on synthetic TIFFs
python scripts/smoke_test.py --pretrained coco   # also verifies COCO weight loading
python scripts/verify_readiness.py          # 17-check pre-training gate (see below)
```

`scripts/smoke_test.py` builds two synthetic specimens in MIDOG++ format and exercises
metadata joining, the region grid, training, sliding-window inference, region inference,
threshold sweeping, matching and counting. Its numbers are not model performance.

---

## Data

**Nothing is downloaded automatically.** The full collection is **65.05 GiB** over 506
files; the frozen cohort needs **7.52 GiB** of that.

```bash
# point at a disk with room (C: has 158 GB free; the project drive does not)
export MIDOG_DATA_ROOT="C:/midogpp_data"

python scripts/download_specimens.py --dry-run     # exact sizes, downloads nothing
python scripts/download_specimens.py               # cohort only
python scripts/download_specimens.py --only target_pool
```

The downloader refuses to start if free space is insufficient, resumes partial files, and
verifies byte-exact sizes. It never fetches specimens outside the frozen cohort.

Metadata is already in `data/metadata/`: the official manifest, the annotation JSON, exact
per-file sizes, and copies of the official `evaluation.py` / `nms_WSI.py` used as references.

---

## Pipeline

Each stage is a separate, config-driven command. Run them in order.

```bash
# 1. regions + physical spacing + normalisation statistics  (needs images)
python scripts/prepare_regions.py

# 2. source-domain RetinaNet baseline  (needs GPU for a real run)
python scripts/train_detector.py --tag source_baseline

# 3. threshold calibration + the five-strategy comparison
python scripts/run_acquisition_experiment.py \
    --checkpoint outputs/runs/<run>/checkpoint.pth --tag pilot

# 4. optional: raw predictions for a role
python scripts/infer_predictions.py \
    --checkpoint outputs/.../checkpoint.pth \
    --roles target_assess --out outputs/predictions/target_assess.json

# 5. demonstration interface
streamlit run frontend/app.py
```

Any setting can be overridden without editing files:

```bash
python scripts/train_detector.py --set training.updates=500 --set training.batch_size=4
python scripts/run_acquisition_experiment.py --checkpoint ... --set acquisition.budget_fraction=0.1
```

Colab: `notebooks/colab_pipeline.ipynb` wraps the same commands.

---

## Repository layout

```
midogcx/                 installable package (all scientific logic)
  data/metadata.py       manifest + annotation JSON, ID normalisation, region labelling
  data/images.py         TIFF reading, physical spacing, tissue masks
  data/patches.py        non-overlapping region grid, physical area accounting
  data/splits.py         cohort building, validation, freezing
  data/dataset.py        torch dataset, box-consistent augmentation, sampling
  models/detector.py     RetinaNet construction, strict checkpoint auditing
  models/error_predictor.py  region features, count-error & loss regressors, OOF fitting
  selection/strategies.py    the five acquisition rules
  selection/budget.py        equal-tissue-area budget enforcement
  inference/sliding_window.py tiled whole-specimen inference
  adaptation/trainer.py  matched fine-tuning for every branch
  evaluation/            NMS, one-to-one point matching, metrics, threshold sweep
  acquisition.py         feature rows, target attachment, predictor fitting
scripts/                 one script per pipeline stage + audit + smoke test
configs/                 base.yaml, cohort_v1.yaml
tests/                   98 tests
frontend/app.py          Streamlit demonstration
docs/                    audits, methodology, evaluation protocol, compute plan
data/metadata/           official manifest, annotations, inventory, audit.json
data/splits/             frozen cohort
outputs/                 runs, experiments, logs (git-ignored)
```

The original brief suggested top-level `preprocessing/`, `models/`, `selection/` etc. They
live inside one package instead, so imports are stable, tests can import the real code, and
Colab only has to install one directory.

---

## Scientific conventions (enforced in code, tested)

| Topic | Decision | Where |
|---|---|---|
| Bounding boxes | **xyxy**, verified against the SQLite export | `data/metadata.py` |
| Category 1 vs 2 | 1 = mitotic figure (label **1**); 2 = curated hard negative, never a target | `models/detector.py` |
| Head size | `num_classes=2` **includes background**; only `cls_logits` is reinitialised from COCO, verified key-by-key | `models/detector.py` |
| Region labelling | annotation **centres**, half-open boxes, each cell in exactly one region | `data/metadata.py` |
| Area budget | unique **physical** mm² of tissue, non-overlapping grid, overshoot skipped | `selection/budget.py` |
| Matching | one-to-one, 7.5 µm, distances in µm from each specimen's own spacing | `evaluation/matching.py` |
| Threshold | strict `score > t`, chosen on calibration specimens only, applied to all methods | `evaluation/evaluate.py` |
| Duplicates | whole-specimen centre-distance NMS (25 px), not per tile | `evaluation/nms.py` |
| Empty cases | metric with a zero denominator is 0.0 (deliberate deviation, documented) | `docs/EVALUATION_PROTOCOL.md` |
| Count error | always reported with TP/FP/FN, because FP/FN cancel | `evaluation/metrics.py` |
| Leakage | error-predictor targets only from specimens the source detector never trained on; target-pool labels never reach a selector | `acquisition.py` |

---

## Reproducibility contract

Every run writes, into one directory:

* the full merged config and its SHA-256 fingerprint,
* the seed, and the checkpoint's own architecture + normalisation + update count,
* the locked threshold and the sweep that produced it,
* each strategy's complete ranking **and** admitted region list with the budget used,
* the predictor's out-of-fold and in-sample quality side by side,
* per-specimen metrics, saved raw predictions, comparison table and paired comparisons.

The same config + the same data therefore reproduces the same numbers, and any claim can be
traced back to a stored artefact rather than a log line.

---

## Limitations you must state in the report

1. The target domain is **single-origin (UMC Utrecht) and single-scanner (Hamamatsu XR)**.
   Tumour-type adaptation is confounded with acquisition; do not claim isolated tumour
   effects or multi-site validation.
2. All 55 specimens are official **training** records. They are development data.
3. Patient identifiers are not published, so patient-level independence cannot be proven.
4. Equal annotated area does **not** mean equal expert annotation time.
5. The error predictor is trained on the source domain. Its transfer to melanoma is exactly
   the open question — predicted high current error does not guarantee annotation value.
6. Two seeds support a feasibility judgement, not a statistical superiority claim.

---

## References

1. Liu, E. et al. *A deep active learning framework for mitotic figure detection with minimal
   manual annotation and labelling.* Histopathology 87:536–547, 2025. doi:10.1111/his.15506
2. Tang, J. et al. *Open-Set Active Learning for Nucleus Detection From the Histopathological
   Images.* IEEE TMI 45(3):1062–1074, 2026. doi:10.1109/TMI.2025.3617073
3. Yoo, D., Kweon, I.S. *Learning Loss for Active Learning.* CVPR 2019, 93–102.
4. Aubreville, M. et al. *A comprehensive multi-domain dataset for mitotic figure detection.*
   Scientific Data 10:484, 2023. doi:10.1038/s41597-023-02327-4
5. de Souza, C. *Metric discordance under MIDOG++ domain shift.* J Pathol Inform 22:100705,
   2026. doi:10.1016/j.jpi.2026.100705
6. Xu, S., Dong, H., Zhang, C. *Exploiting Highly Inconsistent Image Regions for Active
   Semi-Supervised Crowd Counting.* IEEE SPL 33:1346–1350, 2026. doi:10.1109/LSP.2026.3676320
7. Yang, C., Huang, L., Crowley, E.J. *Plug and Play Active Learning for Object Detection.*
   CVPR 2024, 17784–17793.
