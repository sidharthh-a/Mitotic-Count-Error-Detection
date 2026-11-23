"""Generate the tissue-region grid, physical spacing table and normalisation statistics.

    python scripts/prepare_regions.py
    python scripts/prepare_regions.py --only source_train source_calib
    python scripts/prepare_regions.py --check   # verify the frozen grid, write nothing

Outputs:
  data/splits/regions.csv        every candidate region with role and physical tissue area
  data/splits/geometry.csv       per-specimen size, spacing and where the spacing came from
  outputs/metadata/normalization.json   channel mean/std from source training specimens

Regions are a non-overlapping 512x512 grid, so annotated tissue area is additive and the
acquisition budget can be enforced in mm^2 rather than patch counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from midogcx.config import load_config, resolve_data_root, resolve_path, resolve_specimen_path
from midogcx.data.images import image_mean_std, read_geometry, read_rgb, tissue_mask
from midogcx.data.metadata import build_index
from midogcx.data.patches import generate_regions, regions_to_frame
from midogcx.data.splits import load_cohort
from midogcx.utils import get_logger, save_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these cohort roles")
    ap.add_argument(
        "--check",
        action="store_true",
        help="recompute and verify the frozen grid on disk; never writes anything",
    )
    args = ap.parse_args(argv)

    log = get_logger("prepare_regions")
    cfg = load_config(args.config, args.set)
    data_root = resolve_data_root(cfg)
    index = build_index(resolve_path(cfg, "paths.manifest"), resolve_path(cfg, "paths.annotations"))
    cohort = load_cohort(resolve_path(cfg, "paths.cohort"))

    roles = args.only or list(cohort.roles)
    patch = int(cfg["dataset"]["patch_size"])
    min_frac = float(cfg["dataset"]["min_tissue_fraction"])
    downsample = int(cfg["dataset"]["mask_downsample"])

    role_of = {sid: role for role, ids in cohort.roles.items() for sid in ids}
    selected = [sid for role in roles for sid in cohort.roles.get(role, [])]

    missing = [sid for sid in selected if not resolve_specimen_path(data_root, f"{sid:03d}.tiff").exists()]
    if missing:
        log.error(
            "%d specimens are missing under %s (e.g. %s). Run scripts/download_specimens.py first.",
            len(missing),
            data_root,
            [f"{s:03d}.tiff" for s in missing[:5]],
        )
        return 2

    all_regions = []
    geometry_rows = []
    for i, sid in enumerate(selected, 1):
        path = resolve_specimen_path(data_root, f"{sid:03d}.tiff")
        geom = read_geometry(path)
        rgb = read_rgb(path)
        mask, factor = tissue_mask(rgb, downsample=downsample)
        del rgb  # ~350 MB per specimen; do not keep it alive across iterations

        regions = generate_regions(
            index,
            geom,
            mask,
            factor,
            patch_size=patch,
            min_tissue_fraction=min_frac,
            role=role_of[sid],
            slide_id=sid,
        )
        all_regions.extend(regions)
        geometry_rows.append(
            {
                "slide_id": sid,
                "file_name": Path(path).name,
                "width": geom.width,
                "height": geom.height,
                "spacing_x_um": geom.spacing_x_um,
                "spacing_y_um": geom.spacing_y_um,
                "spacing_source": geom.spacing_source,
                "n_regions": len(regions),
                "tissue_area_mm2": sum(r.tissue_area_um2 for r in regions) / 1e6,
            }
        )
        log.info(
            "[%d/%d] %s %s: %d regions, %.2f mm2 tissue, spacing %.4f um/px (%s)",
            i,
            len(selected),
            Path(path).name,
            role_of[sid],
            len(regions),
            geometry_rows[-1]["tissue_area_mm2"],
            geom.spacing_x_um,
            geom.spacing_source,
        )

    regions_path = resolve_path(cfg, "paths.regions")
    geometry_path = regions_path.with_name("geometry.csv")
    norm_path = resolve_path(cfg, "paths.normalization")
    frame = regions_to_frame(all_regions)
    geometry_frame = pd.DataFrame(geometry_rows)

    print("\nRegions per role")
    print(frame.groupby("role").agg(n_regions=("region_id", "size"), tissue_mm2=("tissue_area_um2", lambda s: s.sum() / 1e6), n_positives=("n_positives", "sum")).to_string())

    # Normalisation statistics must come from source training specimens only, so that
    # target-domain pixels never influence preprocessing.
    source_ids = [sid for sid in cohort.roles.get("source_train", []) if sid in selected]
    mean: list[float] | None = None
    std: list[float] | None = None
    if source_ids:
        mean, std = image_mean_std([resolve_specimen_path(data_root, f"{sid:03d}.tiff") for sid in source_ids])
        log.info("normalisation mean=%s std=%s", [round(m, 4) for m in mean], [round(s, 4) for s in std])
    else:
        log.warning("source_train not in the selected roles; normalisation not computed")

    if args.check:
        return _verify_existing(regions_path, geometry_path, norm_path, frame, geometry_frame, mean, std, log)

    regions_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(regions_path, index=False)
    geometry_frame.to_csv(geometry_path, index=False)
    if mean is not None:
        save_json(
            {
                "mean": mean,
                "std": std,
                "mode": "source_train_specimens",
                "n_specimens": len(source_ids),
                "note": "Computed only from source-domain training specimens, never from target data.",
            },
            norm_path,
        )

    spacing = geometry_frame[["slide_id", "spacing_x_um", "spacing_y_um", "spacing_source"]]
    print("\nSpacing sources:", spacing["spacing_source"].value_counts().to_dict())
    return 0


def _verify_existing(regions_path, geometry_path, norm_path, frame, geometry_frame, mean, std, log) -> int:
    """Verify the frozen region grid against a fresh recomputation, writing nothing.

    Regions, spacing and normalisation statistics define every selection and every budget
    downstream, so a remote runtime must confirm them rather than regenerate them: a
    re-derived grid could differ silently under a different image library version and
    invalidate every previously reported number.
    """
    import numpy as np

    for path in (regions_path, geometry_path, norm_path):
        if not Path(path).exists():
            log.error("missing frozen artefact %s; run without --check first", path)
            return 2

    problems: list[str] = []

    stored = pd.read_csv(regions_path)
    if len(stored) != len(frame):
        problems.append(f"regions.csv has {len(stored)} rows, recomputed {len(frame)}")
    else:
        for column in ("slide_id", "x", "y", "width", "height", "tissue_px", "n_positives"):
            if not np.array_equal(stored[column].to_numpy(), frame[column].to_numpy()):
                problems.append(f"regions.csv column {column!r} differs")
        for column in ("tissue_area_um2",):
            if not np.allclose(stored[column].to_numpy(dtype=float), frame[column].to_numpy(dtype=float), rtol=0, atol=1e-6):
                problems.append(f"regions.csv column {column!r} differs")
        if list(stored["role"]) != list(frame["role"]):
            problems.append("regions.csv role assignment differs")

    stored_geo = pd.read_csv(geometry_path)
    if len(stored_geo) != len(geometry_frame):
        problems.append(f"geometry.csv has {len(stored_geo)} rows, recomputed {len(geometry_frame)}")
    else:
        for column in ("slide_id", "width", "height", "n_regions"):
            if not np.array_equal(stored_geo[column].to_numpy(), geometry_frame[column].to_numpy()):
                problems.append(f"geometry.csv column {column!r} differs")
        for column in ("spacing_x_um", "spacing_y_um", "tissue_area_mm2"):
            if not np.allclose(stored_geo[column].to_numpy(dtype=float), geometry_frame[column].to_numpy(dtype=float), rtol=1e-9, atol=1e-9):
                problems.append(f"geometry.csv column {column!r} differs")
        if list(stored_geo["spacing_source"]) != list(geometry_frame["spacing_source"]):
            problems.append("geometry.csv spacing_source differs")

    if mean is not None:
        from midogcx.utils import load_json

        stored_norm = load_json(norm_path)
        if not np.allclose(stored_norm["mean"], mean, rtol=0, atol=1e-9) or not np.allclose(
            stored_norm["std"], std, rtol=0, atol=1e-9
        ):
            problems.append("normalization.json differs")

    if problems:
        for problem in problems:
            log.error("frozen artefact mismatch: %s", problem)
        log.error("refusing to continue: the region grid on disk is not reproducible")
        return 3

    log.info(
        "frozen region grid verified (%d regions, %d specimens, normalisation matches); nothing written",
        len(frame),
        len(geometry_frame),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
