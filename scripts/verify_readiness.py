"""Final pre-training verification pass.

    python scripts/verify_readiness.py
    python scripts/verify_readiness.py --skip-regions   # frozen grid check needs images

Checks, in order:

1. frozen cohort: payload hash, agreement with the human-authored roles file, composition
2. frozen region grid: reproducibility of regions.csv / geometry.csv / normalization.json
   (delegated to ``prepare_regions.py --check``; requires specimen images)
3. detector buffers: evaluation, sliding-window inference, region inference and the
   per-region loss computation must all leave BatchNorm running statistics untouched
4. five acquisition strategies: identical candidate pool, identical physical-area budget,
   no duplicate admission, on the real region grid
5. checkpoint save/reload: exact tensor equality, unchanged trainable pattern, eval mode,
   and no pretrained weights fetched during reload
6. Colab notebook: valid JSON, no broken shell escapes, frozen artefacts only ever
   verified (``--check``) and never regenerated
7. frozen artefacts: byte-identical before and after the whole pass

Exit code 0 only if every executed check passes. Nothing here spends GPU time or writes
into ``data/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from midogcx.config import PROJECT_ROOT, load_config, resolve_path, resolve_specimen_path
from midogcx.data.patches import Region, regions_from_frame
from midogcx.data.splits import load_cohort
from midogcx.inference.sliding_window import predict_array, predict_regions
from midogcx.models.detector import (
    DetectorConfig,
    build_detector,
    load_checkpoint,
    read_checkpoint_meta,
    save_checkpoint,
    set_trainable_backbone_layers,
)
from midogcx.selection.budget import (
    assert_disjoint_regions,
    budget_from_fraction,
    select_under_budget,
)
from midogcx.selection.strategies import STRATEGIES, StrategyContext, rank_regions
from midogcx.utils import load_yaml, sha256_text

FROZEN = [
    "data/splits/cohort_v1.json",
    "data/splits/cohort_v1_summary.json",
    "data/splits/regions.csv",
    "data/splits/geometry.csv",
    "outputs/metadata/normalization.json",
]

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}: {detail}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_frozen() -> dict[str, str]:
    return {rel: file_hash(PROJECT_ROOT / rel) for rel in FROZEN if (PROJECT_ROOT / rel).exists()}


# --------------------------------------------------------------------------------------
# 1. Frozen cohort
# --------------------------------------------------------------------------------------
def check_cohort(cfg: dict) -> None:
    cohort_path = resolve_path(cfg, "paths.cohort")
    payload = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort = load_cohort(cohort_path)  # raises if payload_sha256 does not match

    recorded = payload.get("payload_sha256", "")
    recomputed = sha256_text(str(sorted({k: [int(s) for s in v] for k, v in payload["roles"].items()}.items())))

    roles_yaml = load_yaml(PROJECT_ROOT / "configs" / "cohort_v1.yaml")["roles"]
    from_yaml = {k: [int(s) for s in v] for k, v in roles_yaml.items()}
    from_file = {k: [int(s) for s in v] for k, v in payload["roles"].items()}
    matches_yaml = from_file == from_yaml
    differing = sorted(k for k in set(from_file) | set(from_yaml) if from_file.get(k) != from_yaml.get(k))

    ok = bool(recorded) and recorded == recomputed and matches_yaml
    detail = (
        f"digest {recorded[:12]}…, {len(cohort.roles)} roles, "
        f"{sum(len(v) for v in cohort.roles.values())} specimens, "
        f"{'matches' if matches_yaml else f'DIFFERS from roles yaml in {differing}'}"
    )
    record("frozen cohort", ok, detail)

    expected_sizes = {
        "source_train": 20,
        "source_calib": 5,
        "source_assess": 8,
        "target_pool": 12,
        "target_assess": 10,
    }
    sizes = {k: len(v) for k, v in cohort.roles.items()}
    record("cohort composition", sizes == expected_sizes, f"{sizes}")


# --------------------------------------------------------------------------------------
# 2. Frozen region grid
# --------------------------------------------------------------------------------------
def check_region_grid(cfg: dict, data_root: Path, skip: bool) -> None:
    if skip:
        record("frozen region grid", True, "skipped (--skip-regions)")
        return
    if not any(data_root.glob("*.tiff")):
        record("frozen region grid", True, f"skipped (no images under {data_root})")
        return

    before = snapshot_frozen()
    proc = subprocess.run(
        [sys.executable, "scripts/prepare_regions.py", "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    after = snapshot_frozen()
    unchanged = before == after
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    record(
        "frozen region grid",
        proc.returncode == 0 and unchanged,
        (tail[-1] if tail else f"exit {proc.returncode}") + ("" if unchanged else " [ARTEFACTS CHANGED]"),
    )


# --------------------------------------------------------------------------------------
# 3. BatchNorm buffers during evaluation and inference
# --------------------------------------------------------------------------------------
def _buffer_diff(model) -> list[str]:
    return [
        name
        for name, buf in model.named_buffers()
        if not torch.equal(buf, _BUFFERS[name])
    ]


_BUFFERS: dict[str, torch.Tensor] = {}


class _FakeRegionDataset:
    def __init__(self, items, **kwargs):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        image = torch.full((3, 512, 512), 0.5)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
            "labels": torch.tensor([1]),
            "area": torch.tensor([900.0]),
            "iscrowd": torch.tensor([0]),
        }
        return image, target


def check_buffers() -> None:
    from PIL import Image

    import midogcx.adaptation.trainer as trainer

    model = build_detector(DetectorConfig(pretrained="none"))
    model.eval()
    global _BUFFERS
    _BUFFERS = {name: buf.detach().clone() for name, buf in model.named_buffers()}
    n_buffers = len(_BUFFERS)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # (a) whole-image sliding-window inference
        rgb = np.random.default_rng(0).integers(0, 255, size=(768, 1024, 3), dtype=np.uint8)
        predict_array(model, rgb, batch_size=2, device="cpu")
        changed_eval = _buffer_diff(model)
        record(
            "buffers: sliding-window inference",
            not changed_eval and model.training is False,
            f"{n_buffers} buffers unchanged" if not changed_eval else f"mutated {changed_eval[:3]}",
        )

        # (b) per-region inference
        image_path = tmp_path / "001.tiff"
        Image.fromarray(rgb).save(image_path, format="TIFF")
        probe = [
            Region(f"001_{x}_{y}", 1, x, y, 256, 256, 1000, 15625.0, 0, "target_pool")
            for y in (0, 256)
            for x in (0, 256)
        ]
        predict_regions(model, image_path, probe, batch_size=2, device="cpu")
        changed_regions = _buffer_diff(model)
        record(
            "buffers: region inference",
            not changed_regions and model.training is False,
            f"{len(probe)} regions, buffers unchanged" if not changed_regions else f"mutated {changed_regions[:3]}",
        )

    # (c) per-region loss, which must temporarily enter train mode and fully restore it
    original = trainer.MitosisRegionDataset
    trainer.MitosisRegionDataset = _FakeRegionDataset  # type: ignore[assignment]
    try:
        class _Item:
            def __init__(self, rid: str) -> None:
                self.region_id = rid

        losses = trainer.detector_loss_per_region(
            model,
            [_Item("001_0_0"), _Item("001_512_0")],
            index=None,
            data_root=".",
            normalization={"mean": [0.5] * 3, "std": [0.25] * 3},
            device="cpu",
        )
    finally:
        trainer.MitosisRegionDataset = original  # type: ignore[assignment]

    changed_loss = _buffer_diff(model)
    record(
        "buffers: per-region loss",
        not changed_loss and model.training is False and len(losses) == 2,
        (
            f"{len(losses)} regions, buffers unchanged, mode restored"
            if not changed_loss
            else f"mutated {changed_loss[:3]}"
        ),
    )


# --------------------------------------------------------------------------------------
# 4. Five strategies: identical pool, identical budget
# --------------------------------------------------------------------------------------
class _StubPredictor:
    feature_names = ("proxy",)

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


def check_strategies(cfg: dict) -> None:
    regions_df = pd.read_csv(resolve_path(cfg, "paths.regions"))
    regions = regions_from_frame(regions_df)
    pool = [r for r in regions if r.role == "target_pool"]
    assert_disjoint_regions(pool)
    record("region grid disjoint", True, f"{len(pool)} target_pool regions, no overlap")

    rng = np.random.default_rng(11)
    detections, features = {}, {}
    for region in pool:
        n = int(rng.integers(0, 6))
        scores = rng.uniform(0.3, 0.99, size=n)
        detections[region.region_id] = (rng.uniform(0, 400, size=(n, 2)), scores)
        features[region.region_id] = {"proxy": float(n) + float(rng.uniform(0, 0.1))}

    ctx = StrategyContext(
        detections=detections,
        threshold=0.5,
        seed=int(cfg["project"]["seed"]),
        predictors={"count_error": _StubPredictor(), "learned_loss": _StubPredictor()},
        features=features,
    )
    lookup = {r.region_id: r for r in pool}
    pool_ids = set(lookup)
    fraction = float(cfg["acquisition"]["budget_fraction"])
    largest = max(r.tissue_area_um2 for r in pool)

    used: dict[str, float] = {}
    rows = []
    for name in STRATEGIES:
        ranking = rank_regions(pool, name, ctx)
        ranked_ids = [rid for rid, _ in ranking]
        same_pool = set(ranked_ids) == pool_ids and len(ranked_ids) == len(pool_ids)

        budget = budget_from_fraction(pool, fraction)
        chosen = select_under_budget(lookup, ranking, budget)
        unique = len({r.region_id for r in chosen}) == len(chosen)
        within = budget.used_um2 <= budget.total_um2 + 1e-6
        used[name] = budget.used_um2
        rows.append((name, same_pool, len(chosen), budget.used_um2 / 1e6, budget.utilisation, unique and within))

    spread = max(used.values()) - min(used.values())
    aligned = all(row[1] for row in rows)
    no_dup = all(row[5] for row in rows)
    equal_budget = spread <= largest + 1e-6

    print("\n  strategy        same_pool  selected   mm2      utilisation")
    for name, same_pool, n, mm2, util, ok in rows:
        print(f"  {name:14s}  {'yes' if same_pool else 'NO ':8s}  {n:8d}  {mm2:7.4f}  {util:9.1%}")

    record(
        "strategies: aligned pool",
        aligned,
        f"all {len(STRATEGIES)} strategies ranked the same {len(pool_ids)} regions",
    )
    record(
        "strategies: equal annotation budget",
        equal_budget and no_dup,
        f"spread {spread / 1e6:.6f} mm2 <= one region ({largest / 1e6:.6f} mm2), utilisation "
        f"{min(r[4] for r in rows):.1%}–{max(r[4] for r in rows):.1%}",
    )


# --------------------------------------------------------------------------------------
# 5. Checkpoint save/reload
# --------------------------------------------------------------------------------------
def check_checkpoint(cfg: dict) -> None:
    normalization = json.loads(resolve_path(cfg, "paths.normalization").read_text(encoding="utf-8"))
    detector_cfg = DetectorConfig.from_dict(cfg["model"])

    model = build_detector(DetectorConfig(pretrained="none"))
    set_trainable_backbone_layers(model, detector_cfg.trainable_backbone_layers)
    with torch.no_grad():  # make the weights non-trivial so a silent no-op reload is visible
        for name, param in list(model.named_parameters())[:40]:
            param.add_(0.001 * torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape) % 0.01)

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainable_before = sorted(n for n, p in model.named_parameters() if p.requires_grad)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pth"
        save_checkpoint(
            path,
            model,
            detector_cfg,
            normalization,
            meta={"updates": 123, "tag": "readiness", "seed": int(cfg["project"]["seed"])},
        )
        loaded, payload = load_checkpoint(path, device="cpu")
        meta = read_checkpoint_meta(path)

    after = loaded.state_dict()
    differing = [k for k in before if not torch.equal(before[k], after[k])]
    trainable_after = sorted(n for n, p in loaded.named_parameters() if p.requires_grad)
    record(
        "checkpoint: weights",
        not differing,
        f"{len(before)} tensors identical" if not differing else f"{len(differing)} differ: {differing[:3]}",
    )
    record(
        "checkpoint: trainable pattern",
        trainable_before == trainable_after,
        f"{len(trainable_before)} trainable tensors, identical set",
    )
    record(
        "checkpoint: provenance",
        payload["detector_config"]["pretrained"] == detector_cfg.pretrained
        and meta["meta"].get("updates") == 123
        and [round(v, 9) for v in meta["normalization"]["mean"]]
        == [round(float(v), 9) for v in normalization["mean"]],
        "config, normalisation and meta survived the round trip",
    )
    record(
        "checkpoint: no pretrained fetch on reload",
        loaded._load_report["reinitialised"] == ["all"] and loaded.training is False,
        "rebuilt without weights, overwritten by checkpoint, eval mode",
    )


# --------------------------------------------------------------------------------------
# 6. Colab notebook
# --------------------------------------------------------------------------------------
def check_notebook() -> None:
    path = PROJECT_ROOT / "notebooks" / "colab_pipeline.ipynb"
    if not path.exists():
        record("colab notebook", False, "missing notebooks/colab_pipeline.ipynb")
        return
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    stray = [i for i, src in enumerate(code_cells) if "\\" in src]
    record("notebook: valid JSON", nb.get("nbformat") == 4, f"{len(nb['cells'])} cells, nbformat 4")

    frozen_calls = []
    for src in code_cells:
        for line in src.splitlines():
            stripped = line.strip().lstrip("!").strip()
            for script in ("build_cohort.py", "prepare_regions.py"):
                if script in stripped:
                    frozen_calls.append((script, "--check" in stripped, stripped))
    unguarded = [c for c in frozen_calls if not c[1]]
    record(
        "notebook: frozen artefacts verified not regenerated",
        not unguarded and bool(frozen_calls),
        (
            f"{len(frozen_calls)} guarded call(s): "
            + ", ".join(sorted({c[0] for c in frozen_calls}))
            if not unguarded
            else f"unguarded: {[c[2] for c in unguarded]}"
        ),
    )
    record("notebook: no broken shell escapes", not stray, "all code cells free of backslashes" if not stray else f"cells {stray}")


# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--skip-regions", action="store_true", help="skip the frozen grid recheck")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    from midogcx.config import resolve_data_root

    before = snapshot_frozen()
    print(f"frozen artefacts under observation: {len(before)}")

    check_cohort(cfg)
    check_region_grid(cfg, resolve_data_root(cfg), args.skip_regions)
    check_buffers()
    check_strategies(cfg)
    check_checkpoint(cfg)
    check_notebook()

    after = snapshot_frozen()
    record(
        "frozen artefacts unchanged",
        before == after,
        f"{len(before)} files byte-identical" if before == after else "MODIFIED during verification",
    )

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 72)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
        print("=" * 72)
        return 1
    print("ALL CHECKS PASSED — cleared for source-baseline training")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
