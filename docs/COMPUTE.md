# Compute plan

Measured facts, then what they imply. No cloud cost or runtime figures are invented here.

## Measured on this machine

| Quantity | Value | How |
|---|---|---|
| Device | Windows 11, i5-1155G7, 4C/8T, 7.7 GiB RAM | `docs/ENVIRONMENT_AUDIT.md` |
| GPU | Intel Iris Xe, no CUDA | `torch.cuda.is_available() → False` |
| torch | 2.7.0+cpu, torchvision 0.22.0+cpu | runtime |
| Training step, batch 1, 512×512, R50-FPN | **2.4–2.8 s/update** | `scripts/smoke_test.py` |
| Whole-specimen inference, 2048×1536 | 14.8 s | smoke test |
| Decode of a real 7215×5412 TIFF | ~117 Mpx ≈ 352 MB as uint8 | geometry + file sizes |

The smoke test's training numbers come from randomly initialised weights on synthetic
images; they measure *throughput*, not accuracy. Real batches are usually 2–4, which costs
more per update but more per second of useful gradient.

## What that implies

**Source baseline (500 updates, batch 1):** ≈ 21 min on CPU.
**Six branches with a 500-update adaptation budget:** ≈ 2 h training.
**Assessment inference, 10 specimens × 6 models:** each specimen is ~12× the smoke-test
image, so ≈ 3 min per whole-specimen pass → ≈ 3 h.
**Region inference for features (22 melanoma + 13 breast specimens):** ≈ 6 h.

Total for one seed ≈ **10–12 hours of continuous CPU**. That is an overnight run on a
laptop that will thermally throttle, and it leaves no headroom to fix mistakes.

**Recommendation: run training and inference on Colab.** The local machine keeps doing what
it is good at — audits, region generation, selection logic, evaluation of saved
predictions, interface, tests.

## Colab plan

`notebooks/colab_pipeline.ipynb` runs:

1. install dependencies (CUDA torch wheel),
2. mount Drive, point `MIDOG_DATA_ROOT` at the data,
3. download the frozen cohort on the Colab machine (space permitting) or read it from Drive,
4. `prepare_regions.py`, `train_detector.py`,
5. `run_acquisition_experiment.py`,
6. zip `outputs/` back to Drive.

Practical notes:

* Colab disks are ephemeral. Keep images and `outputs/` on Drive, and copy only what a cell
  needs to local disk if I/O becomes the bottleneck.
* Roughly 7.5 GiB of images plus checkpoints (~146 MB each) must fit the Drive quota.
* Session limits mean the source baseline and the branches may need separate sessions. Every
  checkpoint carries its own config, normalisation and update count, so resuming is a matter
  of pointing at the saved checkpoint rather than reconstructing state.
* If Drive I/O makes whole-specimen inference too slow, run inference once and keep the raw
  prediction JSON (a few MB), then re-evaluate locally. Thresholds are applied offline by
  design, so no re-inference is needed to change them.

## If only CPU is available

A reduced, still-honest fallback: 150–200 updates per branch, 6 assessment specimens, one
seed, reporting wider uncertainty intervals and an explicit statement that the budgets are
smaller than the protocol's. That is a feasibility study, not the full comparison, and must
be labelled as such.
