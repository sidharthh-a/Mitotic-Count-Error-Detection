"""Run the controlled acquisition experiment.

For every strategy, in the same order and under the same settings:

1. rank the melanoma-pool candidate regions (no target labels),
2. admit regions until the tissue-area budget is spent,
3. fine-tune a fresh copy of the *same* source checkpoint on those regions,
4. evaluate on independent melanoma assessment specimens,
5. save selections, checkpoint, per-specimen metrics and a paired comparison.

A ``source_only`` control (no adaptation) is always evaluated, because a strategy can only
be interpreted relative to what the unadapted detector already achieves.

    python scripts/run_acquisition_experiment.py --checkpoint <source ckpt> --tag pilot

Requires specimen images and a trained source checkpoint; both need GPU time for a real
run (see README, `docs/COMPUTE.md`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from midogcx.acquisition import build_feature_rows, fit_predictors
from midogcx.adaptation.trainer import TrainConfig, detector_loss_per_region, train_detector
from midogcx.config import (
    config_fingerprint,
    load_config,
    resolve_data_root,
    resolve_path,
    resolve_specimen_path,
)
from midogcx.data.dataset import items_from_regions
from midogcx.data.metadata import build_index
from midogcx.data.patches import regions_from_frame
from midogcx.evaluation.evaluate import (
    SlidePrediction,
    evaluate_specimens,
    save_predictions,
    select_threshold,
)
from midogcx.evaluation.metrics import paired_comparison, metrics_table
from midogcx.evaluation.nms import nms_keep
from midogcx.inference.sliding_window import predict_regions, predict_slide
from midogcx.models.detector import DetectorConfig, load_checkpoint
from midogcx.selection.budget import (
    assert_disjoint_regions,
    budget_from_fraction,
    select_under_budget,
)
from midogcx.selection.strategies import StrategyContext, rank_regions, ranking_overlap
from midogcx.utils import create_run_dir, get_logger, load_json, save_json


def _spacing_table(regions_path: Path) -> dict[int, tuple[float, float]]:
    geo = pd.read_csv(regions_path.with_name("geometry.csv"))
    return {
        int(r.slide_id): (float(r.spacing_x_um), float(r.spacing_y_um)) for r in geo.itertuples()
    }


def _thresholds(cfg: dict) -> list[float]:
    s = cfg["evaluation"]["threshold_search"]
    return [round(float(t), 4) for t in np.arange(float(s["start"]), float(s["stop"]) + 1e-9, float(s["step"]))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checkpoint", required=True, help="source-domain checkpoint")
    ap.add_argument("--tag", default="acquisition")
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", help="rank + select only, no fine-tuning")
    ap.add_argument(
        "--stop-after-control",
        action="store_true",
        help="evaluate the unadapted source_only control and stop (baseline gate)",
    )
    args = ap.parse_args(argv)

    log = get_logger("experiment")
    cfg = load_config(args.config, args.set)
    data_root = resolve_data_root(cfg)
    device = args.device or ("cuda" if _cuda() else "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        # Fail before a run directory is created: the failure mode without this is a
        # traceback from inside nn.Module._apply, several seconds after the run starts.
        ap.error(
            f"--device {device} needs a CUDA build of torch; this interpreter has "
            f"torch {torch.__version__} with cuda available: False. "
            f"Pass --device cpu, or run on a GPU runtime."
        )
    index = build_index(resolve_path(cfg, "paths.manifest"), resolve_path(cfg, "paths.annotations"))
    cohort = load_cohort_safe(cfg)
    regions_df = pd.read_csv(resolve_path(cfg, "paths.regions"))
    regions = regions_from_frame(regions_df)
    norm = load_json(resolve_path(cfg, "paths.normalization"))
    spacing = _spacing_table(resolve_path(cfg, "paths.regions"))

    run_dir = create_run_dir(resolve_path(cfg, "project.outputs") / "experiments", args.tag)
    save_json({"config": cfg, "fingerprint": config_fingerprint(cfg), "checkpoint": args.checkpoint}, run_dir / "config.json")
    log.info("run directory: %s (device=%s)", run_dir, device)

    infer_cfg = cfg["inference"]
    eval_cfg = cfg["evaluation"]
    acq_cfg = cfg["acquisition"]
    strategies = args.strategies or list(acq_cfg["strategies"])

    by_role = {role: [r for r in regions if r.role == role] for role in cohort.roles}
    pool = by_role.get("target_pool", [])
    if not pool:
        log.error("no target_pool regions; run scripts/prepare_regions.py first")
        return 2
    # Equal-area budgets are only meaningful if the pool's regions are disjoint.
    assert_disjoint_regions(pool)

    # ---- 1. Source model, and a threshold chosen on calibration specimens only --------
    model, payload = load_checkpoint(args.checkpoint, device=device)
    calib_ids = cohort.roles.get("source_calib", [])
    log.info("calibrating threshold on %d source calibration specimens ...", len(calib_ids))
    calib_preds = _predict_slides(model, calib_ids, data_root, infer_cfg, norm, device, log)
    threshold_info = select_threshold(
        calib_preds,
        calib_ids,
        gt_fn=lambda sid: index.positives(sid),
        spacing=spacing,
        thresholds=_thresholds(cfg),
        nms_radius_px=float(infer_cfg["nms_radius_px"]),
        match_radius_um=float(eval_cfg["match_radius_um"]),
    )
    threshold = float(threshold_info["best_threshold"])
    save_json(threshold_info, run_dir / "threshold.json")
    log.info("locked threshold %.2f (pooled F1 on calibration specimens)", threshold)

    # ---- 2. Error-predictor supervision from an out-of-sample source cohort ------------
    err_roles = list(acq_cfg.get("error_training_roles", ["source_calib", "source_assess"]))
    err_regions = [r for role in err_roles for r in by_role.get(role, [])]
    log.info("fitting acquisition predictors on %s (%d regions)", err_roles, len(err_regions))
    err_features, err_counts, err_detections, err_groups = _region_features(
        model, err_regions, data_root, infer_cfg, norm, device, threshold, log
    )
    regions_by_id = {r.region_id: r for r in err_regions}
    region_losses = detector_loss_per_region(
        model,
        [i for i in items_from_regions(err_regions)],
        index=index,
        data_root=data_root,
        normalization=norm,
        device=device,
    )
    # Groups are derived from the region list itself, not from the order in which
    # per-specimen inference happened to run: GroupKFold must split by specimen, and a
    # misaligned group vector would silently leak specimens across folds.
    err_groups = [r.slide_id for r in err_regions]
    count_model, loss_model, predictor_report = fit_predictors(
        err_features,
        [r.region_id for r in err_regions],
        err_counts,
        regions_by_id,
        region_losses,
        groups=err_groups,
        # The block carries both estimator hyperparameters and the CV fold count;
        # fit_predictors separates them so n_splits never reaches the estimator.
        params=acq_cfg.get("error_predictor", {}),
    )
    count_model.save(run_dir / "predictor_count_error.pkl")
    loss_model.save(run_dir / "predictor_learned_loss.pkl")
    save_json(predictor_report, run_dir / "predictor_report.json")
    log.info(
        "count-error predictor OOF spearman=%.3f mae=%.3f (in-sample spearman=%.3f)",
        predictor_report["count_error_target"].get("oof_spearman", float("nan")),
        predictor_report["count_error_target"].get("oof_mae", float("nan")),
        predictor_report["count_error_target"].get("insample_spearman", float("nan")),
    )

    # ---- 3. Unlabelled target-pool features ------------------------------------------
    pool_features, _pool_counts, pool_detections, _pool_groups = _region_features(
        model, pool, data_root, infer_cfg, norm, device, threshold, log
    )

    # ---- 4. Selections ---------------------------------------------------------------
    ctx = StrategyContext(
        detections=pool_detections,
        threshold=threshold,
        seed=int(cfg["project"]["seed"]),
        uncertainty_delta=float(acq_cfg["uncertainty_delta"]),
        predictors={"count_error": count_model, "learned_loss": loss_model},
        features=pool_features,
    )
    pool_by_id = {r.region_id: r for r in pool}
    rankings, selections, budgets = {}, {}, {}
    for name in strategies:
        ranking = rank_regions(pool, name, ctx)
        budget = budget_from_fraction(pool, float(acq_cfg["budget_fraction"]))
        chosen = select_under_budget(pool_by_id, ranking, budget)
        rankings[name] = ranking
        selections[name] = chosen
        budgets[name] = budget.to_dict()
        (run_dir / "selections").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"region_id": rid, "rank": i + 1, "score": float(score), "selected": rid in budget.admitted}
                for i, (rid, score) in enumerate(ranking)
            ]
        ).to_csv(run_dir / "selections" / f"{name}.csv", index=False)
        save_json(
            {"strategy": name, "threshold": threshold, "budget": budget.to_dict(),
             "selected_regions": [r.to_dict() for r in chosen]},
            run_dir / "selections" / f"{name}.json",
        )
        log.info(
            "%-14s selected %3d regions, %.3f mm2 (%.1f%% of budget), positives revealed=%d",
            name,
            len(chosen),
            budget.used_um2 / 1e6,
            100 * budget.utilisation,
            sum(r.n_positives for r in chosen),
        )

    overlap = {
        a: {b: ranking_overlap(rankings[a], rankings[b], k=20) for b in strategies} for a in strategies
    }
    save_json({"top20_overlap": overlap, "budget": budgets}, run_dir / "ranking_overlap.json")

    if args.dry_run:
        log.info("dry run: stopping before adaptation/evaluation")
        return 0

    # ---- 5. Branches ----------------------------------------------------------------
    assess_ids = cohort.roles.get("target_assess", [])
    gts = {sid: index.positives(sid) for sid in list(assess_ids) + list(calib_ids)}
    results: dict[str, dict] = {}
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    log.info("evaluating source_only control on %d assessment specimens ...", len(assess_ids))
    source_preds = _predict_slides(model, assess_ids, data_root, infer_cfg, norm, device, log)
    results["source_only"] = evaluate_specimens(
        source_preds, assess_ids, gt_fn=lambda s: gts[s], spacing=spacing,
        score_threshold=threshold, nms_radius_px=float(infer_cfg["nms_radius_px"]),
        match_radius_um=float(eval_cfg["match_radius_um"]),
        n_boot=int(eval_cfg["bootstrap_samples"]), seed=int(cfg["project"]["seed"]),
    )
    save_predictions(list(source_preds.values()), metrics_dir / "source_only_predictions.json")
    save_json(results["source_only"], metrics_dir / "source_only.json")

    if args.stop_after_control:
        # Gate before spending GPU time on any adaptation branch: an unadapted control
        # that cannot detect anything makes the acquisition comparison uninterpretable.
        table = metrics_table({"source_only": results["source_only"]})
        table.to_csv(run_dir / "comparison.csv", index=False)
        print("\n" + table.to_string(index=False))
        log.info(
            "baseline gate: wrote source_only results to %s and stopped "
            "(--stop-after-control)",
            run_dir,
        )
        return 0

    branch_updates = int(acq_cfg.get("adaptation_updates", cfg["training"]["updates"]))
    for name in strategies:
        chosen = selections[name]
        if not chosen:
            log.warning("%s selected no regions; skipping", name)
            continue
        log.info("adapting for %s (%d regions, %d updates) ...", name, len(chosen), branch_updates)
        branch_model, _ = load_checkpoint(args.checkpoint, device=device)  # fresh copy
        train_cfg = TrainConfig.from_dict(cfg["training"])
        train_cfg.updates = branch_updates
        train_cfg.seed = int(cfg["project"]["seed"])
        # Honour --device: the config default may name cuda while the caller asked for
        # cpu (or vice versa), and mixed-device branches would not be comparable.
        train_cfg.device = device
        branch_dir = run_dir / "branches" / name
        summary = train_detector(
            branch_model,
            target_items=items_from_regions(chosen),
            index=index,
            data_root=data_root,
            train_cfg=train_cfg,
            run_dir=branch_dir,
            detector_cfg=payload["detector_config"],
            normalization=norm,
            meta={
                "strategy": name,
                "n_selected_regions": len(chosen),
                "budget_used_mm2": budgets[name]["used_mm2"],
                "threshold": threshold,
                "source_checkpoint": str(args.checkpoint),
            },
        )
        preds = _predict_slides(branch_model, assess_ids, data_root, infer_cfg, norm, device, log)
        results[name] = evaluate_specimens(
            preds, assess_ids, gt_fn=lambda s: gts[s], spacing=spacing,
            score_threshold=threshold, nms_radius_px=float(infer_cfg["nms_radius_px"]),
            match_radius_um=float(eval_cfg["match_radius_um"]),
            n_boot=int(eval_cfg["bootstrap_samples"]), seed=int(cfg["project"]["seed"]),
        )
        results[name]["train_summary"] = {k: v for k, v in summary.items() if k != "loss_trace"}
        save_predictions(list(preds.values()), metrics_dir / f"{name}_predictions.json")
        save_json(results[name], metrics_dir / f"{name}.json")

    # ---- 6. Comparison --------------------------------------------------------------
    table = metrics_table(results)
    table.to_csv(run_dir / "comparison.csv", index=False)
    paired = {}
    if "source_only" in results:
        for name, agg in results.items():
            if name == "source_only":
                continue
            paired[f"{name}_vs_source_only"] = paired_comparison(
                _as_metrics(agg), _as_metrics(results["source_only"]),
                n_boot=int(eval_cfg["bootstrap_samples"]), seed=int(cfg["project"]["seed"]),
            )
    save_json(paired, run_dir / "paired_comparisons.json")
    print("\n" + table.to_string(index=False))
    log.info("wrote results to %s", run_dir)
    return 0


def _as_metrics(agg: dict):
    """Rebuild SpecimenMetrics from a saved summary (drop the derived columns)."""
    from midogcx.evaluation.metrics import SpecimenMetrics

    fields = set(SpecimenMetrics.__dataclass_fields__)
    return [SpecimenMetrics(**{k: v for k, v in row.items() if k in fields}) for row in agg["per_specimen"]]


def _predict_slides(model, slide_ids, data_root, infer_cfg, norm, device, log):
    out = {}
    for sid in slide_ids:
        path = resolve_specimen_path(data_root, f"{sid:03d}.tiff")
        pred = predict_slide(
            model, path,
            tile=int(infer_cfg["tile_size"]), stride=int(infer_cfg["stride"]),
            batch_size=int(infer_cfg["batch_size"]), device=device,
            mean=norm["mean"], std=norm["std"], score_floor=float(infer_cfg["score_floor"]),
            nms_radius_px=None,
        )
        out[sid] = pred
    log.info("  predicted %d specimens", len(out))
    return out


def _region_features(model, regions, data_root, infer_cfg, norm, device, threshold, log):
    """Detect inside each region and build unlabelled feature rows.

    Duplicate suppression is applied **once**, here, with no score threshold. Every
    consumer then sees the same deduplicated detections: the locked counts, the density
    baseline and the uncertainty baseline can no longer disagree about how many figures a
    region contains. Filtering by score afterwards is equivalent to filtering first and
    suppressing second, because suppression always visits points in descending score
    order, so a below-threshold point can never displace a point above the threshold.
    """
    features: dict[str, dict] = {}
    counts: dict[str, int] = {}
    detections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    radius = float(infer_cfg["nms_radius_px"])
    per_slide: dict[int, list] = {}
    for r in regions:
        per_slide.setdefault(r.slide_id, []).append(r)

    for sid, slide_regions in per_slide.items():
        path = resolve_specimen_path(data_root, f"{sid:03d}.tiff")
        raw = predict_regions(
            model, path, slide_regions,
            batch_size=int(infer_cfg["batch_size"]), device=device,
            mean=norm["mean"], std=norm["std"], score_floor=float(infer_cfg["score_floor"]),
        )
        kept: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for r in slide_regions:
            points, scores = raw.get(r.region_id, (np.zeros((0, 2)), np.zeros(0)))
            kept[r.region_id] = nms_keep(points, scores, radius=radius, threshold=None)

        f, c = build_feature_rows(slide_regions, kept, threshold, nms_radius_px=radius)
        for r in slide_regions:
            features[r.region_id] = f[r.region_id]
            counts[r.region_id] = c[r.region_id]
            detections[r.region_id] = kept[r.region_id]
    groups = [r.slide_id for r in regions]
    log.info("  features for %d regions across %d specimens", len(features), len(per_slide))
    return features, counts, detections, groups


def load_cohort_safe(cfg):
    from midogcx.data.splits import load_cohort

    return load_cohort(resolve_path(cfg, "paths.cohort"))


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


if __name__ == "__main__":
    sys.exit(main())
