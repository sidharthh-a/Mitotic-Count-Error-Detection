"""Loading, validating and joining the official MIDOG++ manifest and annotation JSON.

Documented conventions (verified against the published files, see docs/DATASET_AUDIT.md):

* ``datasets_xvalidation.csv`` is semicolon-delimited with columns
  ``Slide;Dataset;Tumor;Scanner;Origin;Species``. ``Slide`` is an integer; the TIFF
  filename is the same number zero-padded to three digits (``094.tiff``).
* ``MIDOG++.json`` is COCO-*like* but **bounding boxes are xyxy**, not xywh. Verified:
  ``[4336, 346, 4386, 396]`` is a 50x50 box whose centre matches the SQLite export.
* Category 1 is ``mitotic figure`` (the only positive class). Category 2 is
  ``not mitotic figure``: a curated hard negative, never a second positive target.
* The JSON contains 553 image records. Fifty of them (``151.tiff``..``200.tiff``) are
  absent from the 503-row manifest and carry **no annotations at all**. They are not
  negatives; supervised eligibility is defined by the manifest, and those records are
  excluded rather than silently labelled as background.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_CATEGORY = 1
NEGATIVE_CATEGORY = 2

MANIFEST_COLUMNS = ["slide_id", "official_split", "tumor", "scanner_raw", "origin", "species"]

#: Scanner spellings that are the same physical device.
#:
#: Evidence: ``Hamammatsu XR`` occurs only for human breast cancer (50 slides) and
#: ``Hamamatsu XR`` only for melanoma (49) and neuroendocrine (55); all three are
#: UMC Utrecht. No specimen carries both spellings, and no other scanner model appears
#: in those groups, so the merged label does not blur two distinct devices. The mapping
#: is still recorded explicitly because it is a text-normalisation judgement, not a fact
#: published by the dataset authors.
SCANNER_ALIASES = {"Hamammatsu XR": "Hamamatsu XR"}

#: Tumour-type spelling differences between the manifest and the JSON.
TUMOR_ALIASES = {"canine lymphosarcoma": "canine lymphoma"}


def normalize_scanner(raw: str) -> str:
    return SCANNER_ALIASES.get(raw, raw)


def normalize_tumor(raw: str) -> str:
    return TUMOR_ALIASES.get(raw, raw)


def slide_to_filename(slide_id: int) -> str:
    """``94 -> '094.tiff'``. Never assume the manifest row order equals the ID."""
    return f"{int(slide_id):03d}.tiff"


def filename_to_slide(file_name: str) -> int:
    return int(Path(file_name).stem)


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read the eligibility manifest and normalise columns."""
    man = pd.read_csv(path, sep=";")
    required = {"Slide", "Dataset", "Tumor", "Scanner", "Origin", "Species"}
    missing = required - set(man.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    man = man.rename(
        columns={
            "Slide": "slide_id",
            "Dataset": "official_split",
            "Tumor": "tumor",
            "Scanner": "scanner_raw",
            "Origin": "origin",
            "Species": "species",
        }
    )
    man["slide_id"] = man["slide_id"].astype(int)
    if man["slide_id"].duplicated().any():
        dups = man.loc[man["slide_id"].duplicated(), "slide_id"].tolist()
        raise ValueError(f"manifest has duplicate slide ids: {dups}")
    man["file_name"] = man["slide_id"].map(slide_to_filename)
    man["scanner"] = man["scanner_raw"].map(normalize_scanner)
    man["tumor"] = man["tumor"].map(normalize_tumor)
    man["official_split"] = man["official_split"].str.strip().str.lower()
    return man[MANIFEST_COLUMNS + ["file_name", "scanner"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------------------
@dataclass
class AnnotationTables:
    """Image and annotation tables parsed from ``MIDOG++.json``."""

    images: pd.DataFrame
    annotations: pd.DataFrame
    n_images_in_json: int
    n_annotations: int
    n_positive: int
    n_negative: int


def load_annotations(path: str | Path) -> AnnotationTables:
    """Parse the annotation JSON into tidy tables without dropping any record."""
    import json

    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    images = pd.DataFrame(payload["images"])
    anns = pd.DataFrame(payload["annotations"])

    images["slide_id"] = images["file_name"].map(filename_to_slide)
    images["tumor_type"] = images["tumor_type"].map(normalize_tumor)

    bbox = np.asarray(anns["bbox"].tolist(), dtype=float).reshape(-1, 4)
    anns["x1"], anns["y1"], anns["x2"], anns["y2"] = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    dims = images.set_index("id")[["width", "height"]]
    anns = anns.join(dims, on="image_id")
    anns["slide_id"] = anns["image_id"].map(images.set_index("id")["slide_id"])

    # Centre coordinates. xyxy is the verified convention; centre is robust to boxes
    # that extend past the image edge, which is why point evaluation uses centres.
    anns["cx"] = (anns["x1"] + anns["x2"]) / 2.0
    anns["cy"] = (anns["y1"] + anns["y2"]) / 2.0

    anns["geometry_valid"] = (anns["x2"] > anns["x1"]) & (anns["y2"] > anns["y1"])
    anns["center_in_bounds"] = (
        (anns["cx"] >= 0)
        & (anns["cx"] <= anns["width"])
        & (anns["cy"] >= 0)
        & (anns["cy"] <= anns["height"])
    )
    anns["is_positive"] = anns["category_id"] == POSITIVE_CATEGORY

    return AnnotationTables(
        images=images,
        annotations=anns,
        n_images_in_json=int(len(images)),
        n_annotations=int(len(anns)),
        n_positive=int(anns["is_positive"].sum()),
        n_negative=int((anns["category_id"] == NEGATIVE_CATEGORY).sum()),
    )


# --------------------------------------------------------------------------------------
# Combined index
# --------------------------------------------------------------------------------------
@dataclass
class MidogIndex:
    """Manifest + annotations joined, with supervised eligibility defined by the manifest.

    ``slides`` holds the 503 eligible specimens. ``unannotated_json_only`` records the 50
    extra JSON image records that are deliberately excluded from every experiment.
    ``audit`` stores the numbers that docs/DATASET_AUDIT.md reports.
    """

    slides: pd.DataFrame
    images: pd.DataFrame
    annotations: pd.DataFrame
    unannotated_json_only: pd.DataFrame
    audit: dict = field(default_factory=dict)

    # -- basic queries ---------------------------------------------------------------
    def slide(self, slide_id: int) -> pd.Series:
        matches = self.slides[self.slides["slide_id"] == int(slide_id)]
        if matches.empty:
            raise KeyError(f"slide {slide_id} is not eligible (absent from manifest)")
        return matches.iloc[0]

    @property
    def slide_ids(self) -> list[int]:
        return self.slides["slide_id"].tolist()

    def eligible(self, slide_id: int) -> bool:
        return int(slide_id) in set(self.slides["slide_id"])

    # -- annotations ----------------------------------------------------------------
    def positives(self, slide_id: int) -> np.ndarray:
        """``(N, 2)`` array of positive mitotic-figure centres in image pixels."""
        sub = self.annotations[
            (self.annotations["slide_id"] == int(slide_id)) & self.annotations["is_positive"]
        ]
        return sub[["cx", "cy"]].to_numpy(dtype=float)

    def negatives(self, slide_id: int) -> np.ndarray:
        """``(N, 2)`` centres of curated non-mitotic annotations (hard negatives)."""
        sub = self.annotations[
            (self.annotations["slide_id"] == int(slide_id))
            & (self.annotations["category_id"] == NEGATIVE_CATEGORY)
        ]
        return sub[["cx", "cy"]].to_numpy(dtype=float)

    def positive_boxes(self, slide_id: int) -> np.ndarray:
        """``(N, 4)`` xyxy boxes of positives (may extend past the image edge)."""
        sub = self.annotations[
            (self.annotations["slide_id"] == int(slide_id)) & self.annotations["is_positive"]
        ]
        return sub[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)

    def counts(self, slide_id: int) -> dict[str, int]:
        sub = self.annotations[self.annotations["slide_id"] == int(slide_id)]
        return {
            "positives": int(sub["is_positive"].sum()),
            "negatives": int((sub["category_id"] == NEGATIVE_CATEGORY).sum()),
        }

    def annotations_in_box(self, slide_id: int, x: float, y: float, w: float, h: float) -> dict:
        """Annotations whose **centre** falls inside the box ``[x, x+w) x [y, y+h)``.

        Using the centre is the documented region-labelling rule: a box straddling the
        crop boundary belongs to exactly one region, so region counts never double-count
        a cell.
        """
        sub = self.annotations[self.annotations["slide_id"] == int(slide_id)]
        inside = sub[
            (sub["cx"] >= x) & (sub["cx"] < x + w) & (sub["cy"] >= y) & (sub["cy"] < y + h)
        ]
        pos = inside[inside["is_positive"]]
        neg = inside[inside["category_id"] == NEGATIVE_CATEGORY]
        return {
            "positives": pos[["cx", "cy"]].to_numpy(dtype=float),
            "positive_boxes": pos[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
            "negatives": neg[["cx", "cy"]].to_numpy(dtype=float),
        }

    def tumor_counts(self) -> pd.DataFrame:
        return (
            self.slides.groupby(["species", "tumor", "official_split"])
            .size()
            .rename("n_slides")
            .reset_index()
        )


def build_index(manifest_path: str | Path, annotation_path: str | Path) -> MidogIndex:
    """Load both metadata sources, join them, and compute the audit numbers."""
    manifest = load_manifest(manifest_path)
    tables = load_annotations(annotation_path)

    json_only = tables.images[~tables.images["file_name"].isin(manifest["file_name"])].copy()
    # Deliberately verify rather than assume: these records must carry no annotations.
    json_only_ids = set(json_only["id"].tolist())
    json_only_anns = int(tables.annotations["image_id"].isin(json_only_ids).sum())

    per_slide = (
        tables.annotations.groupby("slide_id")
        .agg(
            n_total=("id", "size"),
            n_positive=("is_positive", "sum"),
            n_out_of_bounds=("center_in_bounds", lambda s: int((~s).sum())),
            n_bad_geometry=("geometry_valid", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    slides = manifest.merge(per_slide, on="slide_id", how="left").fillna(
        {"n_total": 0, "n_positive": 0, "n_out_of_bounds": 0, "n_bad_geometry": 0}
    )
    slides["n_positive"] = slides["n_positive"].astype(int)

    audit = {
        "n_manifest_slides": int(len(manifest)),
        "n_json_images": tables.n_images_in_json,
        "n_json_only_images": int(len(json_only)),
        "n_annotations_on_json_only_images": json_only_anns,
        "n_annotations": tables.n_annotations,
        "n_positive_annotations": tables.n_positive,
        "n_negative_annotations": tables.n_negative,
        "n_slides_without_positive": int((slides["n_positive"] == 0).sum()),
        "n_out_of_bounds_centres": int(slides["n_out_of_bounds"].sum()),
        "n_bad_geometry_boxes": int(slides["n_bad_geometry"].sum()),
        "species_counts": slides["species"].value_counts().to_dict(),
        "scanner_raw_values": sorted(manifest["scanner_raw"].unique().tolist()),
        "scanner_normalized_values": sorted(manifest["scanner"].unique().tolist()),
        "origin_values": sorted(manifest["origin"].unique().tolist()),
        "official_split_counts": manifest["official_split"].value_counts().to_dict(),
    }

    return MidogIndex(
        slides=slides,
        images=tables.images,
        annotations=tables.annotations,
        unannotated_json_only=json_only,
        audit=audit,
    )
