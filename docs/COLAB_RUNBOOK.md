# Colab runbook — source baseline training

Exact steps to train the source-domain detector on a fresh Colab runtime. Everything runs
from the repository's own scripts; the notebook is a driver, not a second implementation.

Verified locally before writing this: 108 tests pass, the synthetic smoke test passes, and
`scripts/verify_readiness.py` reports **17/17 checks passed**.

---

## 0. What must be on Google Drive first

| Path on Drive | Contents | Size |
|---|---|---|
| `MyDrive/midogcx_project/` | this repository (code, configs, metadata, frozen splits) | ~9 MB |
| `MyDrive/midogpp_data/` | the 55 cohort TIFFs, named `003.tiff` … `553.tiff` | ~7.6 GB |

Prepare the project archive on Windows (run in PowerShell or Git Bash):

```powershell
cd D:\projects\Mitotic-Count-Error-Detection
Compress-Archive -Path * -DestinationPath "$env:USERPROFILE\Desktop\midogcx_project.zip" -Force
```

Upload that zip to Drive, then unzip it in Drive so the layout is
`MyDrive/midogcx_project/scripts/...`, `MyDrive/midogcx_project/midogcx/...`.

For the images, upload the folder `C:\midogpp_data` to Drive as `midogpp_data`. If uploading
7.6 GB is impractical, skip it and let cell 9 download directly on Colab (it verifies every
file byte-for-byte and resumes partial transfers).

**Drive quota check:** ~7.6 GB images + ~9 MB code + checkpoints (~150 MB each) + results.
Confirm you have ≥ 10 GB free before starting.

---

## 1. Start the runtime

1. Open <https://colab.research.google.com> → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Upload `notebooks/colab_pipeline.ipynb` (File → Upload notebook), or open it from Drive.

---

## 2. Run the cells in order

The notebook refuses to train without CUDA, and refuses to touch the frozen artefacts.

Cell numbers below are the notebook's own positions (1-based, markdown headers included).

**Runtime > Run all works end to end.** The notebook selects CUDA when present and falls
back to CPU with a printed cost warning, reuses an existing baseline checkpoint instead of
retraining it, and reuses a finished baseline gate. Expect ~1.5 h on a T4 and ~7–8 h on CPU,
where the same steps are only slower, not different.

| Cell | What it does | Expected output | Gate |
|---|---|---|---|
| 2 | GPU identity | `Tesla T4, 15360 MiB` | GPU present |
| 3 | install deps, pick `DEVICE` | `DEVICE = cuda` (or `cpu` + cost warning) | warns on CPU, does not stop |
| 5 | mount Drive, set `MIDOG_DATA_ROOT` | cwd + free space | `midogcx_project` found |
| 6 | `pytest tests/ -q` | `117 passed` | **stop if any fail** |
| 8 | `build_cohort.py --check` | `cohort verified unchanged (…)` | **stop if it errors** |
| 9 | `download_specimens.py --dry-run` | per-file sizes, "can hold" | advisement only |
| 10 | `download_specimens.py` | `already present` × 55, or downloads | skip if images are on Drive |
| 12 | `prepare_regions.py --check` | `2,395 regions, 55 specimens … nothing written` | **stop if it errors** |
| 14 | train the source baseline | loss trace every 20 updates | see §3 |
| 15 | select the newest checkpoint | prints `using: outputs/runs/…/checkpoint.pth` | checkpoint exists |

Cells 17–19 (acquisition experiment) are **not** part of this run. Do not run them until the
baseline gate in cell 17 is reviewed.

If `--check` fails, **do not** re-run `prepare_regions.py` without `--check` on Colab. The
frozen grid on Drive would then differ from the one every budget and selection was computed
against. Stop and report the mismatch instead.

---

## 3. The training command itself

Cell 14 runs exactly this:

```bash
python scripts/train_detector.py --tag source_baseline \
    --set training.device=cuda training.batch_size=4 training.updates=500
```

`--set` accepts several `key=value` pairs in one flag. Useful overrides:

| Override | Why |
|---|---|
| `training.updates=200` | first, cheap throughput probe |
| `training.batch_size=2` | if the T4 runs out of memory at 4 |
| `training.checkpoint_every=100` | survival points across disconnects |
| `project.seed=7` | independent replicate (report separately) |

To run the same thing from a plain Colab terminal cell instead of the notebook:

```python
!cd /content/drive/MyDrive/midogcx_project && \
 MIDOG_DATA_ROOT=/content/drive/MyDrive/midogpp_data \
 python scripts/train_detector.py --tag source_baseline \
   --set training.device=cuda training.batch_size=4 training.updates=500
```

### What to record from this run

`outputs/runs/<run>/train_summary.json` contains:

- `updates` actually completed,
- `elapsed_s` and `seconds_per_update` (**measured**, not estimated),
- `final_loss` and the full `loss_trace`,
- `train_config` (every knob that was in force),
- `checkpoint` path.

`outputs/runs/<run>/checkpoint.pth` carries its own architecture, normalisation statistics
and update count, so the acquisition experiment can reload it without any side channel.

**Do not report the loss trace as accuracy.** It shows training ran; detection quality is
unknown until the baseline gate evaluates on specimens.

### Rough runtime expectation (to be replaced by the measured value)

A T4 typically lands around 0.3–0.6 s/update for RetinaNet-R50-FPN at 512², batch 4, so 500
updates should take a few minutes. This is an expectation, not a measurement — the number to
quote is `seconds_per_update` from `train_summary.json`.

---

## 4. Persist the results

Cell 22 zips the experiment directory back to Drive. For the baseline alone, copy the run
directory:

```python
!cp -r /content/drive/MyDrive/midogcx_project/outputs/runs /content/drive/MyDrive/midogcx_results/
```

Checkpoints are ~150 MB each; results JSON/CSV are kilobytes.

---

## 5. Failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA not available` assertion | CPU runtime selected | Runtime → Change runtime type → T4 GPU |
| `pid ... CUDA out of memory` | batch too large | `training.batch_size=2` |
| Training crawls (many s/update) | reading TIFFs over Drive | `cp -r` the needed files to `/content/`, point `MIDOG_DATA_ROOT` there |
| Runtime disconnected mid-training | Colab session limit | everything is on Drive. **Training cannot be resumed** — no optimizer state is saved — so re-run cell 13 from scratch, with `training.checkpoint_every=100` as survival points, not as resume points |
| `cohort failed its integrity check` | cohort file edited or corrupted | stop; restore the frozen file, do not rebuild |
| `frozen artefact mismatch` | region grid not reproducible | stop; report which column differed |
| pytest failures on Colab | environment differs from local | stop before training; fix the environment, not the tests |

---

## 6. Rules that apply to every Colab run

1. **Never** regenerate the frozen cohort or the frozen region grid on a remote runtime.
   Verify with `--check`.
2. **Never** tune anything on `target_assess`. Threshold selection uses `source_calib` only.
3. **Never** present smoke-test numbers, synthetic numbers or the blueprint's historical
   pilot numbers as results of this study.
4. Report counting MAE **together with** precision, recall, F1 and raw FP/FN — count error
   can look small while false positives and false negatives cancel.
5. Every claim must trace to a saved artefact in `outputs/`, not to a log line.

---

## 7. After the baseline trains

1. Persist the checkpoint and `train_summary.json` to Drive.
2. Run the baseline gate (notebook cell 17):

   ```bash
   python scripts/run_acquisition_experiment.py --checkpoint {SOURCE_CKPT} \
       --tag baseline_gate --device cuda --stop-after-control
   ```

   This evaluates the **unadapted** detector on the 10 melanoma assessment specimens and
   stops. If precision/recall/F1 are near zero there, the acquisition comparison cannot be
   interpreted — fix the baseline first.
3. Only if the gate is sane, run the full experiment (cell 18) and update
   `PROJECT_STATUS.md` with the measured values.

### Re-running the gate (cell 17) after the predictor fix

The first Cell 16 attempt failed with
`HistGradientBoostingRegressor.__init__() got an unexpected keyword argument 'n_splits'`.
That is fixed in `midogcx/models/error_predictor.py`, `midogcx/acquisition.py` and
`scripts/run_acquisition_experiment.py`.

1. Copy the fixed `midogcx/`, `scripts/` and `configs/` from the project archive over the
   copy on Drive (no data or checkpoint needs to move).
2. In Colab, re-run cell 6 (`pytest -q`, expect **117 passed**). If this fails, stop: the
   Drive copy is not the fixed code.
3. Re-run cell 15 to set `SOURCE_CKPT`, then cell 17. The exact command is:

```python
!python scripts/run_acquisition_experiment.py --checkpoint {SOURCE_CKPT} --tag baseline_gate --device cuda --stop-after-control
```

Run it from the project directory (cell 5 already `chdir`s there). Keep it on one line — a
`!` command with a backslash continuation does not work in IPython. Expect a
`predictor_report.json` with `n_splits`, `regressor_params` and out-of-fold scores, then the
`source_only` metrics table.
