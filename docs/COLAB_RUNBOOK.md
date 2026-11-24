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

For the images, upload the folder `C:\midogpp_data` to Drive as `midogpp_data`.
Cell 5 copies them to `/content/midogpp_data` once per session and reads from there: a
region crop still decodes the whole ~145 MiB TIFF, so a 500-update run touches ~140 GiB,
which Drive's FUSE mount cannot serve in reasonable time. If the Drive copy is incomplete,
nothing is staged and cell 10 downloads straight to the local disk instead. If uploading
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

The notebook never demands a GPU (it selects CUDA when present and falls back to CPU with a
printed cost warning), and refuses to touch the frozen artefacts.

Cell numbers below are the notebook's own positions (1-based, markdown headers included).

**Runtime > Run all works end to end.** The notebook selects CUDA when present and falls
back to CPU with a printed cost warning, reuses an existing baseline checkpoint instead of
retraining it, and reuses a finished baseline gate. Expect ~1.5 h on a T4 and ~7–8 h on CPU,
where the same steps are only slower, not different.

Every script is launched through one helper (`run_step`) that streams the script's stdout but
captures its stderr: if a step fails, its own traceback is printed after a
`!!! step failed (exit N)` line. That block — not the surrounding `CalledProcessError` — is
the error to read. Cell 15 also loads the chosen checkpoint before the gate and cell 17
prints the free space results are written to, so a torn checkpoint or a full results disk
fails in seconds with a named file instead of after an hour of inference.

| Cell | What it does | Expected output | Gate |
|---|---|---|---|
| 2 | GPU identity | `Tesla T4, 15360 MiB` | GPU present |
| 3 | install only missing deps, pick `DEVICE` | `DEVICE = cuda` (or `cpu` + cost warning) | warns on CPU, does not stop |
| 5 | mount Drive, stage specimens to local disk, set `MIDOG_DATA_ROOT` | project, 55 tiff, free space | `midogcx_project` found |
| 6 | `pytest tests/ -q` | `117 passed` | **stop if any fail** |
| 8 | `build_cohort.py --check` | `cohort verified unchanged (…)` | **stop if it errors** |
| 9 | `download_specimens.py --dry-run` | per-file sizes, "can hold" | advisement only |
| 10 | `download_specimens.py` | `already present` × 55, or downloads | skip if images are on Drive |
| 12 | `prepare_regions.py --check` | `2,395 regions, 55 specimens … nothing written` | **stop if it errors** |
| 14 | train the source baseline | loss trace every 20 updates | see §3 |
| 15 | select the newest checkpoint | prints `using: outputs/runs/…/checkpoint.pth` | checkpoint exists |

Cell 17 is the baseline gate: it evaluates the unadapted control and prints the P/R/F1 /
TP/FP/FN/MAE table. Review that table before cell 18 spends time on the five adaptation
branches — an unadapted control that detects nothing makes the comparison uninterpretable.

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

Cell 23 copies the experiment directory's small artefacts (JSON/CSV/pickles) to Drive and
skips checkpoints. For the baseline alone, copy the run directory:

```python
!cp -r /content/drive/MyDrive/midogcx_project/outputs/runs /content/drive/MyDrive/midogcx_results/
```

Checkpoints are ~150 MB each; results JSON/CSV are kilobytes.

---

## 5. Failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA not available` assertion | CPU runtime selected | Runtime → Change runtime type → T4 GPU |
| `Invalid device string: 'cuda training.batch_size=4 …'` | several `--set key=value` tokens joined into one argv item, so argparse stores the whole string as one value | pass each override as its own argument; the notebook's `train_command()` helper does this |
| `stale or partial project copy` from cell 5 | the Drive copy differs from this repository (a changed, missing or leftover file) | the message lists the offending files; re-upload the repository, then re-run cell 5. Regenerate the manifest first if the repository itself changed: `python scripts/write_notebook_manifest.py` |
| `TypeError: HistGradientBoostingRegressor.__init__() got an unexpected keyword argument 'n_splits'` | the Drive copy of `midogcx/` predates the CV-split fix, so the config block's `n_splits` reaches sklearn | prevented by cell 5's manifest check; if it still appears, the cell was skipped — re-upload `midogcx/`, `scripts/`, `configs/` and `tests/`, then re-run from cell 5 |
| `CUDA out of memory` at batch 4 | T4 memory limit | the notebook retries once at batch 2 automatically; a manual run needs `training.batch_size=2` |
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
`scripts/run_acquisition_experiment.py`. Cell 5 verifies every file the run depends on against a
sha256 manifest — `scripts/write_notebook_manifest.py` rebuilds that manifest — so a stale
Drive copy is named, file by file, before training starts. `scripts/verify_readiness.py` runs
that manifest's `--check` on every pass, so drift is caught locally instead of in Colab; a code
change needs `python scripts/write_notebook_manifest.py`, a documentation change does not
(`docs/` and the top-level `*.md` are not part of the manifest).

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
