"""Verify one detector checkpoint end to end, in a single command.

    python scripts/verify_checkpoint.py
    python scripts/verify_checkpoint.py --checkpoint outputs/runs/source_baseline/checkpoint.pth
    python scripts/verify_checkpoint.py --checkpoint CKPT --probe 3
    python scripts/verify_checkpoint.py --checkpoint CKPT --json outputs/runs/checkpoint_report.json

Checks, in order:

1. file: exists, size, sha256 (recorded so a report identifies the exact bytes verified)
2. layout: the payload is a dict carrying ``model_state``, ``detector_config``,
   ``normalization`` and ``meta`` — a whole-module pickle or an old-style ``state_dict``
   dump fails here instead of at some later stage
3. tensors: count, parameter total, dtypes, and no NaN/Inf in any weight
4. reload: ``load_checkpoint`` rebuilds the architecture from the checkpoint's own config
   (no pretrained fetch), validates the classification-head shape and loads with
   ``strict=True``; the model comes back in eval mode with the expected trainable set
5. normalisation: mean/std equal the frozen ``outputs/metadata/normalization.json``
6. detector config: equal to the shipped ``model:`` section, so the run cannot have
   silently changed architecture or class convention
7. meta: ``role`` is a real cohort role, ``n_regions`` equals the frozen grid's count for
   that role, and ``updates``/``config_fingerprint`` are present
8. fingerprint reproducibility: the recorded ``config_fingerprint`` is recomputed from the
   shipped config, and from any ``--set`` overrides supplied with this run; at least one
   must reproduce it, and the report says which settings do
9. batchnorm statistics are not at their initialisation values (``running_mean`` all zero,
   ``running_var`` all one), which a saved-but-never-updated model would show
10. optional loss probe (``--probe N``): detection loss on held-out regions from a role the
   checkpoint did not train on, compared against a freshly initialised model of the same
   architecture. This is a sanity check that the weights carry signal — it is **not** a
   precision/recall/MAE measurement and must never be reported as one.

Exit code 0 only if every executed check passes. Nothing here writes into ``data/`` and
nothing needs a GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from midogcx.adaptation.trainer import detector_loss_per_region
from midogcx.config import config_fingerprint, load_config, resolve_data_root, resolve_path
from midogcx.data.dataset import items_from_regions
from midogcx.data.metadata import build_index
from midogcx.data.patches import regions_from_frame
from midogcx.data.splits import load_cohort
from midogcx.models.detector import (
    DetectorConfig,
    build_detector,
    load_checkpoint,
)
from midogcx.utils import load_json, sha256_file

RESULTS: list[tuple[str, bool, str]] = []
REQUIRED_KEYS = ("model_state", "detector_config", "normalization", "meta")
# RetinaNet's default anchor grid: 5 scales x 3 aspect ratios.
ANCHORS_PER_LOCATION = 9


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}: {detail}")


def newest_checkpoint(pattern: str) -> Path | None:
    paths = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(paths[0]) if paths else None


def pick_probe_items(regions, role: str, n: int):
    """Held-out regions from ``role``, preferring those that contain at least one mitosis.

    An empty region alone can pass with a collapsed detector (nothing to find), so the
    probe keeps positives in the sample whenever the role has them.
    """
    pool = [item for item in items_from_regions(regions) if item.role == role]
    positives = [item for item in pool if item.has_positive]
    empties = [item for item in pool if not item.has_positive]
    n_positive = min(len(positives), max(1, (n + 1) // 2))
    chosen = positives[:n_positive] + empties[: n - n_positive]
    return chosen[:n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default=None, help="checkpoint path (default: newest under outputs/runs)")
    ap.add_argument("--config", default=None, help="config file (default: configs/base.yaml)")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="overrides used for the run, to reproduce its fingerprint")
    ap.add_argument("--probe", type=int, default=0, help="score N held-out regions against a random-init model (default 0 = skip)")
    ap.add_argument("--probe-role", default=None, help="role to probe (default: an assessment role the checkpoint did not train on)")
    ap.add_argument("--probe-root", default=None, help="data root for the probe (default: resolved from config/env)")
    ap.add_argument("--device", default="cpu", help="device for the reload (default cpu)")
    ap.add_argument("--json", default=None, help="write the full report as JSON to this path")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    outputs = Path(resolve_path(cfg, "project.outputs"))
    ckpt = Path(args.checkpoint) if args.checkpoint else newest_checkpoint(str(outputs / "runs" / "*" / "checkpoint.pth"))

    # 1. file ---------------------------------------------------------------------------
    if ckpt is None or not ckpt.exists():
        record("checkpoint: file", False, f"no checkpoint found (looked for {outputs / 'runs' / '*' / 'checkpoint.pth'})")
        return _finish(None, cfg, args)
    size_mb = ckpt.stat().st_size / 1e6
    digest = sha256_file(ckpt)
    record("checkpoint: file", True, f"{ckpt} — {size_mb:.1f} MB, sha256 {digest[:16]}…")

    # 2. layout -------------------------------------------------------------------------
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is a hard stop
        record("checkpoint: layout", False, f"could not load: {type(exc).__name__}: {exc}")
        return _finish(ckpt, cfg, args)
    if not isinstance(payload, dict):
        record("checkpoint: layout", False, f"payload is {type(payload).__name__}, not a dict (saved whole model instead of payload?)")
        return _finish(ckpt, cfg, args)
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    record("checkpoint: layout", not missing, "keys " + ", ".join(sorted(payload)) + (f" — missing {missing}" if missing else ""))
    if missing:
        return _finish(ckpt, cfg, args)

    state = payload["model_state"]
    meta = payload.get("meta") or {}

    # 3. tensors ------------------------------------------------------------------------
    non_tensor = [k for k, v in state.items() if not torch.is_tensor(v)]
    bad = [k for k, v in state.items() if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any())]
    params = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    dtypes = sorted({str(v.dtype) for v in state.values() if torch.is_tensor(v)})
    record(
        "checkpoint: tensors",
        not non_tensor and not bad,
        f"{len(state)} tensors, {params / 1e6:.1f} M params, dtypes {dtypes}"
        + (f"; non-tensor entries: {non_tensor[:3]}" if non_tensor else "")
        + (f"; NaN/Inf in {bad[:3]}" if bad else "; no NaN/Inf"),
    )

    # 4. reload -------------------------------------------------------------------------
    try:
        model, _ = load_checkpoint(ckpt, device=args.device)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        record(
            "checkpoint: reload",
            not model.training and bool(trainable),
            f"strict load OK, head shape checked, eval mode, {len(trainable)} trainable tensors",
        )
    except Exception as exc:  # noqa: BLE001 - a shape/architecture mismatch must not be swallowed
        record("checkpoint: reload", False, f"{type(exc).__name__}: {exc}")
        return _finish(ckpt, cfg, args)

    # 5. normalisation ------------------------------------------------------------------
    frozen = load_json(resolve_path(cfg, "paths.normalization"))
    ck_norm = payload["normalization"]
    norm_ok = all(
        len(ck_norm.get(k, [])) == len(frozen[k]) and all(abs(a - b) < 1e-12 for a, b in zip(ck_norm[k], frozen[k]))
        for k in ("mean", "std")
    )
    record(
        "checkpoint: normalisation",
        norm_ok,
        "matches frozen normalization.json" if norm_ok else f"{ck_norm} != {[frozen['mean'], frozen['std']]}",
    )

    # 6. detector config ----------------------------------------------------------------
    ck_det = payload["detector_config"]
    cfg_det = cfg["model"]
    same_cfg = DetectorConfig.from_dict(ck_det).__dict__ == DetectorConfig.from_dict(cfg_det).__dict__
    head_ok = True
    head_key = "head.classification_head.cls_logits.weight"
    if head_key in state:
        expected = DetectorConfig.from_dict(ck_det).num_classes * ANCHORS_PER_LOCATION
        head_ok = int(state[head_key].shape[0]) == expected
    record(
        "checkpoint: detector config",
        same_cfg and head_ok,
        f"{ck_det.get('arch')} pretrained={ck_det.get('pretrained')} "
        f"layers={ck_det.get('trainable_backbone_layers')} num_classes={ck_det.get('num_classes')}; "
        + ("matches shipped config" if same_cfg else "DIFFERS from shipped config")
        + ("" if head_ok else "; head width inconsistent with num_classes"),
    )

    # 7. meta vs frozen cohort/grid -----------------------------------------------------
    problems: list[str] = []
    role = meta.get("role")
    try:
        cohort = load_cohort(resolve_path(cfg, "paths.cohort"))
        regions = regions_from_frame(pd.read_csv(resolve_path(cfg, "paths.regions")))
    except Exception as exc:  # noqa: BLE001 - frozen artefacts may be missing on a bare checkout
        cohort = None
        regions = None
        problems.append(f"could not read frozen cohort/grid: {type(exc).__name__}")
    if cohort is not None and role not in (cohort.roles or {}):
        problems.append(f"role {role!r} is not a cohort role")
    if regions is not None and role:
        # regions_from_frame returns Region objects, so count them directly.
        actual = sum(1 for region in regions if region.role == role)
        if int(meta.get("n_regions", -1)) != actual:
            problems.append(f"n_regions {meta.get('n_regions')} != {actual} in the frozen grid")
    for key in ("updates", "config_fingerprint"):
        if not meta.get(key):
            problems.append(f"meta.{key} missing")
    record(
        "checkpoint: meta",
        not problems,
        f"role={role} regions={meta.get('n_regions')} updates={meta.get('updates')}"
        + ("" if not problems else f" — {'; '.join(problems)}"),
    )

    # 8. fingerprint reproducibility ----------------------------------------------------
    target = meta.get("config_fingerprint", "")
    candidates: list[tuple[str, dict]] = [("shipped config", cfg)]
    if args.overrides:
        try:
            candidates.append(("--set " + " ".join(args.overrides), load_config(args.config, args.overrides)))
        except Exception as exc:  # noqa: BLE001 - a bad --set is a user error worth reporting
            record("checkpoint: fingerprint", False, f"could not apply --set: {type(exc).__name__}: {exc}")
            return _finish(ckpt, cfg, args)
    matches = [name for name, candidate in candidates if config_fingerprint(candidate) == target]
    table = "; ".join(f"{name} -> {config_fingerprint(candidate)[:16]}" for name, candidate in candidates)
    record(
        "checkpoint: fingerprint",
        bool(matches),
        (f"reproduced by: {', '.join(matches)}" if matches else "NOT reproduced by any supplied config")
        + f" (recorded {target[:16]}…; {table})",
    )

    # 9. batchnorm statistics versus initialisation -------------------------------------
    bn_means = [buf for name, buf in model.named_buffers() if name.endswith("running_mean")]
    bn_vars = [buf for name, buf in model.named_buffers() if name.endswith("running_var")]
    untouched = bool(bn_means) and all(torch.count_nonzero(b) == 0 for b in bn_means) and all(
        torch.allclose(b, torch.ones_like(b)) for b in bn_vars
    )
    record(
        "checkpoint: batchnorm updated",
        not untouched,
        "running stats are not at initialisation values"
        if not untouched
        else "running_mean all zero and running_var all one — this model may never have run a forward pass",
    )

    # 10. optional loss probe -----------------------------------------------------------
    if args.probe > 0:
        probe_role = args.probe_role
        if probe_role is None and cohort is not None:
            held_out = [r for r in ("source_assess", "target_assess") if r in (cohort.roles or {}) and r != role]
            probe_role = held_out[0] if held_out else None
        if probe_role is None or regions is None:
            record("checkpoint: loss probe", False, "no held-out role available to probe")
        else:
            try:
                root = args.probe_root or resolve_data_root(cfg)
                index = build_index(resolve_path(cfg, "paths.manifest"), resolve_path(cfg, "paths.annotations"))
                probe = pick_probe_items(regions, probe_role, args.probe)
                trained_losses = detector_loss_per_region(model, probe, index, root, frozen, device=args.device)
                fresh = build_detector(replace(DetectorConfig.from_dict(ck_det), pretrained="none"))
                fresh_losses = detector_loss_per_region(fresh, probe, index, root, frozen, device=args.device)
                t_mean = statistics.mean(trained_losses[i.region_id] for i in probe)
                f_mean = statistics.mean(fresh_losses[i.region_id] for i in probe)
                record(
                    "checkpoint: loss probe",
                    t_mean < f_mean,
                    f"{len(probe)} held-out {probe_role} regions "
                    f"({sum(i.has_positive for i in probe)} with mitoses): "
                    f"trained {t_mean:.3f} vs random-init {f_mean:.3f}"
                    + ("" if t_mean < f_mean else " — trained weights are NOT better than random")
                    + " (sanity check only, not a metric)",
                )
            except Exception as exc:  # noqa: BLE001 - probe needs images; absence is a skip-worthy failure
                record("checkpoint: loss probe", False, f"{type(exc).__name__}: {exc}")

    return _finish(ckpt, cfg, args, extra={"meta": meta, "sha256": digest, "size_bytes": ckpt.stat().st_size})


def _finish(ckpt: Path | None, cfg: dict, args, extra: dict | None = None) -> int:
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print()
    print("=" * 72)
    if failed:
        print(f"{passed}/{len(RESULTS)} checks passed — FAILED: {', '.join(failed)}")
    else:
        print(f"{len(RESULTS)}/{len(RESULTS)} checks passed — checkpoint verified")
    print("=" * 72)

    if args.json:
        report = {
            "checkpoint": str(ckpt) if ckpt else None,
            "config": {"fingerprint": config_fingerprint(cfg)},
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS],
            "passed": passed,
            "total": len(RESULTS),
            **(extra or {}),
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"report written: {out}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
