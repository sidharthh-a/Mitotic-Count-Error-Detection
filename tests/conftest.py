"""Shared fixtures. Unit tests use tiny synthetic metadata; one integration test uses the
real downloaded MIDOG++ metadata and is skipped when it is absent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_DIR = PROJECT_ROOT / "data" / "metadata"
MANIFEST = META_DIR / "datasets_xvalidation.csv"
ANNOTATIONS = META_DIR / "MIDOG++.json"


@pytest.fixture(scope="session")
def real_metadata_paths() -> tuple[Path, Path] | None:
    if MANIFEST.exists() and ANNOTATIONS.exists():
        return MANIFEST, ANNOTATIONS
    return None


@pytest.fixture(scope="session")
def real_metadata_available(real_metadata_paths) -> bool:
    return real_metadata_paths is not None


@pytest.fixture
def synthetic_metadata(tmp_path: Path) -> tuple[Path, Path]:
    """Two-slide synthetic dataset exercising the documented conventions.

    Slide 10 (breast, train): one 50x50 positive at x1=100,y1=200 and one non-mitotic box.
    Slide 20 (melanoma, train): one positive whose centre is inside the image but whose
    box extends past the right edge, plus one positive fully inside.
    """
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "Slide;Dataset;Tumor;Scanner;Origin;Species\n"
        "10;train;human breast cancer;Hamammatsu XR;UMC Utrecht;Human\n"
        "20;train;human melanoma;Hamamatsu XR;UMC Utrecht;Human\n",
        encoding="utf-8",
    )

    def ann(i, image_id, bbox, cat):
        return {"id": i, "image_id": image_id, "category_id": cat, "bbox": bbox, "labels": [cat, cat]}

    payload = {
        "info": {"version": "test"},
        "licenses": [],
        "images": [
            {"license": 1, "file_name": "010.tiff", "id": 10, "width": 1000, "height": 800,
             "tumor_type": "human breast cancer"},
            {"license": 1, "file_name": "020.tiff", "id": 20, "width": 1000, "height": 800,
             "tumor_type": "human melanoma"},
            {"license": 1, "file_name": "151.tiff", "id": 151, "width": 1000, "height": 800,
             "tumor_type": "human melanoma"},
        ],
        "categories": [
            {"id": 1, "name": "mitotic figure"},
            {"id": 2, "name": "not mitotic figure"},
        ],
        "annotations": [
            ann(1, 10, [100, 200, 150, 250], 1),
            ann(2, 10, [300, 300, 350, 350], 2),
            ann(3, 20, [950, 100, 1005, 150], 1),
            ann(4, 20, [500, 500, 550, 550], 1),
        ],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, annotations
