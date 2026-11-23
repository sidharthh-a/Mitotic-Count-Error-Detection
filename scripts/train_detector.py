"""Train the source-domain RetinaNet baseline.

    python scripts/train_detector.py --set training.updates=500 --set training.device=cuda

Runs from ``data/splits/regions.csv`` + ``outputs/metadata/normalization.json``, so it is
identical on the local CPU (smoke test / tiny run) and on Colab (real run). The checkpoint
records its own architecture, class convention, normalisation and update count; nothing is
inferred later from a filename.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from midogcx.adaptation.trainer import TrainConfig, train_detector
from midogcx.config import config_fingerprint, load_config, resolve_data_root, resolve_path
from midogcx.data.dataset import items_from_regions
from midogcx.data.metadata import build_index
from midogcx.data.patches import regions_from_frame
from midogcx.models.detector import DetectorConfig, build_detector
from midogcx.utils import create_run_dir, get_logger, load_json, save_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--role", default="source_train", help="cohort role used for fitting")
    ap.add_argument("--tag", default="source_baseline")
    args = ap.parse_args(argv)

    log = get_logger("train")
    cfg = load_config(args.config, args.set)
    data_root = resolve_data_root(cfg)
    index = build_index(resolve_path(cfg, "paths.manifest"), resolve_path(cfg, "paths.annotations"))
    regions = regions_from_frame(pd.read_csv(resolve_path(cfg, "paths.regions")))
    items = [i for i in items_from_regions(regions) if i.role == args.role]
    if not items:
        log.error("no regions with role %r; run scripts/prepare_regions.py first", args.role)
        return 2

    normalization = load_json(resolve_path(cfg, "paths.normalization"))
    train_cfg = TrainConfig.from_dict(cfg["training"])
    train_cfg.seed = int(cfg["project"]["seed"])

    run_dir = create_run_dir(resolve_path(cfg, "project.outputs") / "runs", args.tag, train_cfg.seed)
    save_json({"config": cfg, "fingerprint": config_fingerprint(cfg)}, run_dir / "config.json")

    log.info("building detector (%s, pretrained=%s) ...", cfg["model"]["arch"], cfg["model"]["pretrained"])
    model = build_detector(DetectorConfig.from_dict(cfg["model"]))
    log.info("initialisation report: %s", getattr(model, "_load_report", {}))

    summary = train_detector(
        model,
        target_items=items,
        index=index,
        data_root=data_root,
        train_cfg=train_cfg,
        run_dir=run_dir,
        detector_cfg=cfg["model"],
        normalization=normalization,
        meta={
            "role": args.role,
            "n_regions": len(items),
            "config_fingerprint": config_fingerprint(cfg),
            "purpose": "source-domain baseline detector",
        },
    )
    log.info(
        "finished: %d updates in %.1fs (%.2fs/update), checkpoint %s",
        summary["updates"],
        summary["elapsed_s"],
        summary["seconds_per_update"],
        summary["checkpoint"],
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "loss_trace"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
