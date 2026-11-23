"""Region generation and area-accounting tests."""

from __future__ import annotations

import numpy as np
import pytest

from midogcx.data.images import SlideGeometry, crop_tissue_fraction
from midogcx.data.metadata import build_index
from midogcx.data.patches import generate_regions, region_id, regions_from_frame, regions_to_frame


@pytest.fixture
def tiny_index(synthetic_metadata):
    manifest, annotations = synthetic_metadata
    return build_index(manifest, annotations)


def _geometry(width=1000, height=800, sx=0.25, sy=0.25):
    return SlideGeometry(
        path="010.tiff",
        width=width,
        height=height,
        spacing_x_um=sx,
        spacing_y_um=sy,
        spacing_source="test",
    )


def test_regions_are_a_non_overlapping_grid(tiny_index):
    mask = np.ones((800 // 4, 1000 // 4), dtype=bool)
    regions = generate_regions(tiny_index, _geometry(), mask, factor=4, patch_size=512, slide_id=10)
    assert len(regions) == 4  # 2 x 2 over 1000x800 with 512 tiles
    covered = sum(r.width * r.height for r in regions)
    assert covered == 1000 * 800, "regions must tile the specimen without overlap"
    ids = {r.region_id for r in regions}
    assert len(ids) == len(regions)


def test_region_area_is_physical_not_pixel_count(tiny_index):
    mask = np.ones((200, 250), dtype=bool)
    regions = generate_regions(
        tiny_index, _geometry(width=1000, height=800), mask, factor=4, patch_size=512, slide_id=10
    )
    r = regions[0]
    assert r.tissue_px == r.width * r.height
    assert r.tissue_area_um2 == pytest.approx(r.tissue_px * 0.25 * 0.25)


def test_anisotropic_spacing_changes_physical_area(tiny_index):
    mask = np.ones((200, 250), dtype=bool)
    iso = generate_regions(tiny_index, _geometry(), mask, 4, 512, slide_id=10)[0]
    aniso = generate_regions(
        tiny_index, _geometry(sx=0.25, sy=0.5), mask, 4, 512, slide_id=10
    )[0]
    assert aniso.tissue_area_um2 == pytest.approx(iso.tissue_area_um2 * 2)


def test_background_regions_are_skipped(tiny_index):
    mask = np.zeros((800 // 4, 1000 // 4), dtype=bool)
    mask[: 512 // 4, : 512 // 4] = True  # only the top-left region is tissue
    regions = generate_regions(tiny_index, _geometry(), mask, 4, 512, min_tissue_fraction=0.5)
    assert len(regions) == 1
    assert (regions[0].x, regions[0].y) == (0, 0)


def test_positive_counts_are_assigned_to_exactly_one_region(tiny_index):
    mask = np.ones((200, 250), dtype=bool)
    regions = generate_regions(tiny_index, _geometry(), mask, 4, 512, slide_id=10)
    assert sum(r.n_positives for r in regions) == 1  # slide 10 has one positive at (125,225)


def test_region_id_format():
    assert region_id(94, 512, 1024) == "094_512_1024"


def test_region_frame_roundtrip(tiny_index):
    mask = np.ones((200, 250), dtype=bool)
    regions = generate_regions(tiny_index, _geometry(), mask, 4, 512, slide_id=10, role="source_train")
    frame = regions_to_frame(regions)
    restored = regions_from_frame(frame)
    assert [r.region_id for r in restored] == [r.region_id for r in regions]
    assert restored[0].tissue_area_um2 == pytest.approx(regions[0].tissue_area_um2)
    assert restored[0].role == "source_train"


def test_crop_tissue_fraction_handles_edges():
    mask = np.ones((10, 10), dtype=bool)
    assert crop_tissue_fraction(mask, 4, 0, 0, 40, 40) == pytest.approx(1.0)
    assert crop_tissue_fraction(mask, 4, 2000, 2000, 40, 40) == 0.0


def test_zero_patch_size_rejected(tiny_index):
    with pytest.raises(ValueError):
        generate_regions(tiny_index, _geometry(), np.ones((10, 10), bool), 4, patch_size=0)
