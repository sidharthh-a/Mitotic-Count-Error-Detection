"""End-to-end smoke test on synthetic data (no MIDOG++ images required).

    python scripts/smoke_test.py                # fast: randomly initialised detector
    python scripts/smoke_test.py --pretrained coco   # also verifies COCO weight loading

It creates two small synthetic specimens with a manifest and annotation JSON in exactly
the MIDOG++ format (semicolon CSV, xyxy boxes, categories 1/2), then runs the real code
paths: metadata join, region grid, dataset, one or two training updates, sliding-window
inference, region inference, threshold sweep, matching and counting.

Purpose: prove the machinery executes and measure CPU throughput. It produces **no
scientific result** and its numbers must never be reported as model performance.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from midogcx.acquisition import build_feature_rows, fit_predictors
from midogcx.adaptation.trainer import TrainConfig, detector_loss_per_region, train_detector
from midogcx.data.dataset import items_from_regions
from midogcx.data.images import image_mean_std, read_geometry, read_rgb, tissue_mask
from midogcx.data.metadata import build_index
from midogcx.data.patches import generate_regions
from midogcx.data.splits import build_cohort
from midogcx.evaluation.evaluate import evaluate_specimens, select_threshold
from midogcx.inference.sliding_window import predict_regions, predict_slide
from midogcx.models.detector import DetectorConfig, build_detector
from midogcx.selection.budget import budget_from_fraction, select_under_budget
from midogcx.selection.strategies import StrategyContext, rank_regions

W, H = 2048, 1536
SPACING_UM = 0.25
DPI = int(25400 / SPACING_UM)  # PIL writes ResolutionUnit=inch


def make_specimen(path: Path, rng: np.random.Generator, n_cells: int) -> list[list[float]]:
    """White background with stained blobs; small dark 'mitotic' dots with xyxy boxes."""
    import cv2
    from PIL import Image

    img = np.full((H, W, 3), 245, dtype=np.uint8)
    for _ in range(140):
        centre = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        axes = (int(rng.integers(15, 60)), int(rng.integers(15, 60)))
        angle = float(rng.uniform(0, 180))
        colour = tuple(int(c) for c in rng.integers(120, 220, size=3))
        cv2.ellipse(img, centre, axes, angle, 0, 360, colour, -1)

    boxes = []
    for _ in range(n_cells):
        x = int(rng.integers(60, W - 60))
        y = int(rng.integers(60, H - 60))
        cv2.circle(img, (x, y), 9, (70, 40, 110), -1)
        boxes.append([x - 25, y - 25, x + 25, y + 25])
    Image.fromarray(img).save(path, dpi=(DPI, DPI), compression="tiff_deflate")
    return boxes


def make_dataset(root: Path, rng: np.random.Generator) -> tuple[Path, Path]:
    """Two specimens: 010 breast (source) and 505 melanoma (target), MIDOG++ schema."""
    slides = [(10, "human breast cancer", "human breast cancer"), (505, "human melanoma", "human melanoma")]
    manifest_lines = ["Slide;Dataset;Tumor;Scanner;Origin;Species"]
    images, annotations = [], []
    ann_id = 1
    for sid, tumor, json_tumor in slides:
        name = f"{sid:03d}.tiff"
        boxes = make_specimen(root / name, rng, n_cells=18)
        manifest_lines.append(
            f"{sid};train;{tumor};Hamamatsu XR;UMC Utrecht;Human"
        )
        images.append(
            {"license": 1, "file_name": name, "id": sid, "width": W, "height": H, "tumor_type": json_tumor}
        )
        for i, box in enumerate(boxes):
            annotations.append(
                {"id": ann_id, "image_id": sid, "category_id": 1, "bbox": box, "labels": [1, 1]}
            )
            ann_id += 1
        # One curated non-mitotic annotation per specimen (category 2).
        annotations.append(
            {"id": ann_id, "image_id": sid, "category_id": 2,
             "bbox": [400, 400, 450, 450], "labels": [2, 2]}
        )
        ann_id += 1

    manifest = root / "manifest.csv"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    payload = {
        "info": {"version": "smoke"},
        "licenses": [],
        "images": images,
        "categories": [{"id": 1, "name": "mitotic figure"}, {"id": 2, "name": "not mitotic figure"}],
        "annotations": annotations,
    }
    ann_path = root / "annotations.json"
    ann_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, ann_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained", default="none", choices=["none", "imagenet_backbone", "coco"])
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = ap.parse_args(argv)

    t0 = time.time()
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="midogcx_smoke_"))
    print(f"workdir: {tmp}")

    manifest, ann_path = make_dataset(tmp, rng)
    index = build_index(manifest, ann_path)
    assert len(index.slide_ids) == 2, index.slide_ids
    assert index.audit["n_positive_annotations"] == 36
    print(f"[ok] metadata: {index.audit['n_positive_annotations']} positives, "
          f"{index.audit['n_negative_annotations']} hard negatives")

    geom = read_geometry(tmp / "010.tiff")
    print(f"[ok] geometry: {geom.width}x{geom.height}, spacing {geom.spacing_x_um:.4f} um/px "
          f"from {geom.spacing_source}")
    assert abs(geom.spacing_x_um - SPACING_UM) < 1e-6

    regions = []
    for sid in (10, 505):
        path = tmp / f"{sid:03d}.tiff"
        rgb = read_rgb(path)
        mask, factor = tissue_mask(rgb, downsample=8)
        role = "source_train" if sid == 10 else "target_pool"
        regions += generate_regions(index, read_geometry(path), mask, factor, 512, 0.5, role, sid)
    print(f"[ok] regions: {len(regions)} candidates, "
          f"{sum(r.n_positives for r in regions)} labelled positives inside regions")
    assert len(regions) > 0 and sum(r.n_positives for r in regions) == 36

    cohort = build_cohort(
        index, manifest,
        {"source_train": [10], "target_pool": [505]},
        role_spec={"source_train": {"tumor": "human breast cancer", "n": 1},
                   "target_pool": {"tumor": "human melanoma", "n": 1}},
        allow_historical_pilot=True,
    )
    assert cohort.role_of(505) == "target_pool"
    print("[ok] cohort validated")

    mean, std = image_mean_std([tmp / "010.tiff"])
    norm = {"mean": mean, "std": std}

    print(f"[...] building detector (pretrained={args.pretrained})")
    det_cfg = DetectorConfig(pretrained=args.pretrained, min_size=512, max_size=512)
    model = build_detector(det_cfg)
    report = getattr(model, "_load_report", {})
    print(f"[ok] detector built; re-initialised weights: {report.get('reinitialised')}")
    assert not report.get("unexpected_keys"), report

    items = items_from_regions([r for r in regions if r.role == "source_train"])
    train_cfg = TrainConfig(
        updates=args.updates, batch_size=1, samples_per_epoch=4, log_every=1,
        warmup_updates=0, device="cpu", seed=0, num_workers=0,
    )
    summary = train_detector(
        model, items, index=index, data_root=tmp, train_cfg=train_cfg,
        run_dir=tmp / "run", detector_cfg=det_cfg, normalization=norm,
        meta={"purpose": "smoke test only"},
    )
    print(f"[ok] trained {summary['updates']} updates in {summary['elapsed_s']}s "
          f"({summary['seconds_per_update']}s/update); loss {summary['final_loss']}")
    assert summary["updates"] == args.updates
    assert (tmp / "run" / "checkpoint.pth").exists()

    start = time.time()
    pred = predict_slide(model, tmp / "010.tiff", tile=512, stride=460, batch_size=2,
                         device="cpu", mean=mean, std=std, score_floor=0.01, nms_radius_px=None)
    print(f"[ok] whole-slide inference: {len(pred.scores)} raw detections in {time.time() - start:.1f}s")

    target_regions = [r for r in regions if r.role == "target_pool"]
    det = predict_regions(model, tmp / "505.tiff", target_regions, device="cpu",
                          mean=mean, std=std, score_floor=0.01)
    assert len(det) == len(target_regions)
    print(f"[ok] region inference: {len(det)} regions")

    spacing = {10: (geom.spacing_x_um, geom.spacing_y_um), 505: (geom.spacing_x_um, geom.spacing_y_um)}
    preds = {10: pred}
    gts = {10: index.positives(10)}
    sweep = select_threshold(preds, [10], gt_fn=lambda s: gts[s], spacing=spacing,
                             thresholds=[0.05, 0.1, 0.2], nms_radius_px=25)
    print(f"[ok] threshold sweep: best {sweep['best_threshold']} over {sweep['search_range']}")

    metrics = evaluate_specimens(preds, [10], gt_fn=lambda s: gts[s], spacing=spacing,
                                 score_threshold=sweep["best_threshold"], nms_radius_px=25, n_boot=20)
    print(f"[ok] evaluation: TP={metrics['pooled']['tp']} FP={metrics['pooled']['fp']} "
          f"FN={metrics['pooled']['fn']} MAE={metrics['counting']['mae']}")

    features, counts = build_feature_rows(target_regions, det, sweep["best_threshold"])
    assert len(features) == len(target_regions)
    ctx = StrategyContext(detections=det, threshold=sweep["best_threshold"], seed=0)
    for name in ("random", "density", "uncertainty"):
        ranking = rank_regions(target_regions, name, ctx)
        budget = budget_from_fraction(target_regions, 0.2)
        chosen = select_under_budget({r.region_id: r for r in target_regions}, ranking, budget)
        print(f"[ok] strategy {name:12s}: ranked {len(ranking)}, selected {len(chosen)} "
              f"({budget.used_um2 / 1e6:.3f} mm2)")

    losses = detector_loss_per_region(
        model, items_from_regions(target_regions), index=index, data_root=tmp,
        normalization=norm, device="cpu",
    )
    assert len(losses) == len(target_regions)
    print(f"[ok] per-region detector loss: mean {np.mean(list(losses.values())):.4f}")

    if len(target_regions) >= 10:
        groups = [r.slide_id for r in target_regions]
        try:
            cm, lm, rep = fit_predictors(
                features, [r.region_id for r in target_regions], counts,
                {r.region_id: r for r in target_regions}, losses, groups,
                params={"max_iter": 20}, n_splits=2,
            )
            print(f"[ok] predictors fitted on {rep['n_regions']} regions")
        except ValueError as exc:
            print(f"[skip] predictor fit not possible on this tiny synthetic set: {exc}")
    else:
        print("[skip] predictor fit (need >=10 target regions)")

    print(f"\nSMOKE TEST PASSED in {time.time() - t0:.1f}s")
    print("Reminder: synthetic data, randomly initialised weights. No scientific result.")
    if not args.keep:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
