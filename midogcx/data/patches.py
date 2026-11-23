"""Tissue-region generation and region bookkeeping.

Regions are a **non-overlapping** grid of ``patch_size`` squares per specimen. The grid
is non-overlapping on purpose: it makes "unique annotated tissue area" additive without
needing overlap bookkeeping, which the research protocol requires for an area-matched
acquisition budget (equal patch counts are not equal area).

A region's supervised content is defined by annotation **centres** inside it, so every
cell belongs to exactly one region. Annotations outside the image are not dropped from
point evaluation, but a region only records the labels whose centres it contains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .images import SlideGeometry, crop_tissue_fraction
from .metadata import MidogIndex


@dataclass(frozen=True)
class Region:
    region_id: str
    slide_id: int
    x: int
    y: int
    width: int
    height: int
    tissue_px: int
    tissue_area_um2: float
    n_positives: int
    role: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def area_mm2(self) -> float:
        return self.tissue_area_um2 / 1e6


def region_id(slide_id: int, x: int, y: int) -> str:
    return f"{int(slide_id):03d}_{int(x)}_{int(y)}"


def generate_regions(
    index: MidogIndex,
    geometry: SlideGeometry,
    mask: np.ndarray,
    factor: int,
    patch_size: int = 512,
    min_tissue_fraction: float = 0.5,
    role: str = "unknown",
    slide_id: int | None = None,
) -> list[Region]:
    """Build the non-overlapping region grid for one specimen.

    ``mask``/``factor`` come from :func:`midogcx.data.images.tissue_mask`. Regions whose
    tissue fraction is below ``min_tissue_fraction`` are skipped: they are background
    and would let a budget be spent on glass.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    sid = int(slide_id if slide_id is not None else Path(geometry.path).stem)
    area_per_px = geometry.area_um2_per_px

    regions: list[Region] = []
    for y in range(0, geometry.height, patch_size):
        for x in range(0, geometry.width, patch_size):
            w = min(patch_size, geometry.width - x)
            h = min(patch_size, geometry.height - y)
            frac = crop_tissue_fraction(mask, factor, x, y, w, h)
            if frac < min_tissue_fraction:
                continue

            tissue_px = int(round(frac * w * h))
            labels = index.annotations_in_box(sid, x, y, w, h)
            regions.append(
                Region(
                    region_id=region_id(sid, x, y),
                    slide_id=sid,
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    tissue_px=tissue_px,
                    tissue_area_um2=float(tissue_px * area_per_px),
                    n_positives=int(len(labels["positives"])),
                    role=role,
                )
            )
    return regions


def region_labels(index: MidogIndex, region: Region) -> dict:
    """Ground-truth content of a region (positives and hard negatives, image coords)."""
    return index.annotations_in_box(region.slide_id, region.x, region.y, region.width, region.height)


def regions_to_frame(regions: list[Region]):
    import pandas as pd

    return pd.DataFrame([r.to_dict() for r in regions])


def regions_from_frame(df) -> list[Region]:
    """Rebuild ``Region`` objects from a saved CSV/DataFrame."""
    cols = [
        "region_id",
        "slide_id",
        "x",
        "y",
        "width",
        "height",
        "tissue_px",
        "tissue_area_um2",
        "n_positives",
        "role",
    ]
    out = []
    for _, row in df.iterrows():
        out.append(
            Region(
                region_id=str(row["region_id"]),
                slide_id=int(row["slide_id"]),
                x=int(row["x"]),
                y=int(row["y"]),
                width=int(row["width"]),
                height=int(row["height"]),
                tissue_px=int(row["tissue_px"]),
                tissue_area_um2=float(row["tissue_area_um2"]),
                n_positives=int(row["n_positives"]),
                role=str(row.get("role", "unknown")),
            )
        )
    return out
