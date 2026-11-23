"""Run sliding-window inference with a checkpoint and save raw predictions.

    python scripts/infer_predictions.py --checkpoint outputs/runs/xxx/checkpoint.pth \
        --roles source_calib source_assess target_assess

Predictions are stored with scores only (no threshold applied), so evaluation can apply
any threshold later without re-running inference. That is what keeps threshold selection on
calibration specimens honest and cheap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from midogcx.config import load_config, resolve_data_root, resolve_path, resolve_specimen_path
from midogcx.data.splits import load_cohort
from midogcx.evaluation.evaluate import save_predictions
from midogcx.inference.sliding_window import predict_slide
from midogcx.models.detector import load_checkpoint
from midogcx.utils import get_logger, save_json

import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--roles", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    log = get_logger("infer")
    cfg = load_config(args.config, args.set)
    data_root = resolve_data_root(cfg)
    cohort = load_cohort(resolve_path(cfg, "paths.cohort"))
    device = args.device or ("cuda" if _cuda() else "cpu")

    model, payload = load_checkpoint(args.checkpoint, device=device)
    norm = payload["normalization"]
    log.info("loaded %s (normalisation from checkpoint)", args.checkpoint)

    slide_ids = [sid for role in args.roles for sid in cohort.roles.get(role, [])]
    if not slide_ids:
        log.error("no specimens for roles %s", args.roles)
        return 2

    infer_cfg = cfg["inference"]
    predictions = []
    timings = {}
    for i, sid in enumerate(slide_ids, 1):
        path = resolve_specimen_path(data_root, f"{sid:03d}.tiff")
        if not path.exists():
            log.error("missing specimen %s", path)
            return 3
        start = time.time()
        pred = predict_slide(
            model,
            path,
            tile=int(infer_cfg["tile_size"]),
            stride=int(infer_cfg["stride"]),
            batch_size=int(infer_cfg["batch_size"]),
            device=device,
            mean=norm["mean"],
            std=norm["std"],
            score_floor=float(infer_cfg["score_floor"]),
            nms_radius_px=None,  # keep raw; thresholds/NMS applied at evaluation time
        )
        timings[str(sid)] = round(time.time() - start, 2)
        predictions.append(pred)
        log.info("[%d/%d] %03d: %d raw detections (%.1fs)", i, len(slide_ids), sid, len(pred.scores), timings[str(sid)])

    out = Path(args.out)
    save_predictions(predictions, out)
    save_json(
        {"checkpoint": str(args.checkpoint), "roles": args.roles, "timings_s": timings},
        out.with_name(out.stem + "_meta.json"),
    )
    log.info("wrote %s", out)
    return 0


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


if __name__ == "__main__":
    sys.exit(main())
