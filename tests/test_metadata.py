"""Annotation parsing, ID normalisation and metadata integrity tests."""

from __future__ import annotations

import numpy as np
import pytest

from midogcx.data.metadata import (
    POSITIVE_CATEGORY,
    build_index,
    filename_to_slide,
    load_annotations,
    load_manifest,
    normalize_scanner,
    slide_to_filename,
)


def test_slide_filename_roundtrip_and_zero_padding():
    assert slide_to_filename(94) == "094.tiff"
    assert slide_to_filename(3) == "003.tiff"
    assert slide_to_filename(505) == "505.tiff"
    assert filename_to_slide("094.tiff") == 94


def test_scanner_alias_only_typo_is_normalised():
    # The documented alias must map; unrelated spellings must not be touched.
    assert normalize_scanner("Hamammatsu XR") == "Hamamatsu XR"
    assert normalize_scanner("Hamamatsu S360") == "Hamamatsu S360"
    assert normalize_scanner("Aperio CS2") == "Aperio CS2"


def test_manifest_loading_derives_filenames(synthetic_metadata):
    manifest, _ = synthetic_metadata
    man = load_manifest(manifest)
    assert len(man) == 2
    assert man["file_name"].tolist() == ["010.tiff", "020.tiff"]
    assert set(man["scanner"]) == {"Hamamatsu XR"}


def test_bbox_is_xyxy_and_centre_is_derived(synthetic_metadata):
    _, annotations = synthetic_metadata
    tables = load_annotations(annotations)
    first = tables.annotations.iloc[0]
    # [100, 200, 150, 250] is a 50x50 box: centre (125, 225), NOT (100+25*?, ...) xywh.
    assert first["x1"] == 100 and first["x2"] == 150
    assert first["cx"] == pytest.approx(125.0)
    assert first["cy"] == pytest.approx(225.0)


def test_positive_and_negative_categories_are_not_confused(synthetic_metadata):
    _, annotations = synthetic_metadata
    tables = load_annotations(annotations)
    positives = tables.annotations[tables.annotations["is_positive"]]
    assert len(positives) == 3
    assert tables.n_positive == 3 and tables.n_negative == 1
    assert set(positives["category_id"]) == {POSITIVE_CATEGORY}


def test_boundary_crossing_positive_keeps_valid_centre(synthetic_metadata):
    _, annotations = synthetic_metadata
    tables = load_annotations(annotations)
    edge = tables.annotations[tables.annotations["id"] == 3].iloc[0]
    # Box [980,100,1030,150] on a 1000 px wide image extends past the right edge.
    assert edge["x2"] > 1000
    assert bool(edge["center_in_bounds"]) is True
    assert bool(edge["geometry_valid"]) is True


def test_index_excludes_unannotated_json_only_records(synthetic_metadata):
    manifest, annotations = synthetic_metadata
    index = build_index(manifest, annotations)
    assert sorted(index.slide_ids) == [10, 20]
    assert index.audit["n_json_only_images"] == 1
    assert index.audit["n_annotations_on_json_only_images"] == 0
    # 151.tiff must never be treated as an eligible (negative) specimen.
    assert not index.eligible(151)


def test_annotations_in_box_uses_centre_and_is_half_open(synthetic_metadata):
    manifest, annotations = synthetic_metadata
    index = build_index(manifest, annotations)
    # Region [0,0,125,225): the positive centre is exactly at x=125 -> excluded.
    left = index.annotations_in_box(10, 0, 0, 125, 225)
    assert len(left["positives"]) == 0
    # Region [0,0,126,226) contains it.
    right = index.annotations_in_box(10, 0, 0, 126, 226)
    assert len(right["positives"]) == 1
    assert right["positives"][0] == pytest.approx([125.0, 225.0])


def test_counts_report_positives_and_negatives(synthetic_metadata):
    manifest, annotations = synthetic_metadata
    index = build_index(manifest, annotations)
    assert index.counts(10) == {"positives": 1, "negatives": 1}
    assert index.positives(20).shape == (2, 2)


def test_real_metadata_matches_published_counts(real_metadata_paths):
    if real_metadata_paths is None:
        pytest.skip("MIDOG++ metadata not downloaded")
    manifest_path, annotation_path = real_metadata_paths
    index = build_index(manifest_path, annotation_path)
    audit = index.audit
    assert audit["n_manifest_slides"] == 503
    assert audit["n_json_images"] == 553
    assert audit["n_json_only_images"] == 50
    assert audit["n_annotations_on_json_only_images"] == 0
    assert audit["n_positive_annotations"] == 11937
    assert audit["n_negative_annotations"] == 14349
    assert audit["species_counts"] == {"Human": 254, "Canine": 249}
    # All 503 manifest specimens must be present in the annotation JSON.
    assert set(index.slides["file_name"]) <= set(index.images["file_name"])
