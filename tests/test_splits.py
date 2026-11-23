"""Cohort validation and freezing tests."""

from __future__ import annotations

import json

import pytest

from midogcx.data.metadata import build_index
from midogcx.data.splits import (
    BLUEPRINT_COHORT,
    HISTORICAL_PILOT_SLIDES,
    build_cohort,
    load_cohort,
    save_cohort,
    validate_cohort,
)


@pytest.fixture
def tiny_index(synthetic_metadata):
    manifest, annotations = synthetic_metadata
    return build_index(manifest, annotations)


def _spec(**kwargs):
    base = {
        "source_train": {"tumor": "human breast cancer", "n": 1},
        "target_pool": {"tumor": "human melanoma", "n": 1},
    }
    base.update(kwargs)
    return base


def test_valid_cohort_passes(tiny_index):
    summary = validate_cohort(
        tiny_index,
        {"source_train": [10], "target_pool": [20]},
        role_spec=_spec(),
    )
    assert summary["source_train"]["n_slides"] == 1
    assert summary["target_pool"]["tumors"] == {"human melanoma": 1}


def test_overlapping_roles_are_rejected(tiny_index):
    with pytest.raises(ValueError, match="appears in both"):
        validate_cohort(
            tiny_index,
            {"source_train": [10], "target_pool": [10]},
            role_spec=_spec(n=2),
        )


def test_unknown_slide_is_rejected(tiny_index):
    with pytest.raises(ValueError, match="not in the official manifest"):
        validate_cohort(
            tiny_index,
            {"source_train": [10, 999]},
            role_spec={"source_train": {"tumor": "human breast cancer", "n": 2}},
        )


def test_wrong_tumour_is_rejected(tiny_index):
    with pytest.raises(ValueError, match="are not 'human melanoma'"):
        validate_cohort(
            tiny_index,
            {"target_pool": [10]},
            role_spec={"target_pool": {"tumor": "human melanoma", "n": 1}},
        )


def test_duplicate_ids_within_a_role_are_rejected(tiny_index):
    with pytest.raises(ValueError, match="duplicate"):
        validate_cohort(
            tiny_index,
            {"source_train": [10, 10]},
            role_spec={"source_train": {"tumor": "human breast cancer", "n": 2}},
        )


def test_historical_pilot_specimens_are_blocked(tiny_index, monkeypatch):
    import midogcx.data.splits as splits_mod

    monkeypatch.setattr(splits_mod, "HISTORICAL_PILOT_SLIDES", {10})
    with pytest.raises(ValueError, match="historical pilot"):
        validate_cohort(
            tiny_index,
            {"source_train": [10]},
            role_spec={"source_train": {"tumor": "human breast cancer", "n": 1}},
        )


def test_freeze_and_reload_roundtrip(tiny_index, tmp_path, synthetic_metadata):
    manifest, _ = synthetic_metadata
    cohort = build_cohort(
        tiny_index,
        manifest,
        {"source_train": [10], "target_pool": [20]},
        name="unit_test",
        seed=7,
        role_spec=_spec(),
    )
    path = tmp_path / "cohort.json"
    digest = save_cohort(cohort, path)
    assert len(digest) == 64

    restored = load_cohort(path)
    assert restored.roles == cohort.roles
    assert restored.seed == 7
    assert restored.role_of(20) == "target_pool"
    assert restored.role_of(10) == "source_train"
    assert restored.role_of(999) is None


def test_tampered_cohort_is_rejected(tiny_index, tmp_path, synthetic_metadata):
    """A cohort whose roles no longer match its recorded hash must not load.

    Everything downstream assumes one immutable split; silently loading a modified file
    could move assessment specimens into training without any error surfacing.
    """
    manifest, _ = synthetic_metadata
    cohort = build_cohort(
        tiny_index,
        manifest,
        {"source_train": [10], "target_pool": [20]},
        name="unit_test",
        seed=7,
        role_spec=_spec(),
    )
    path = tmp_path / "cohort.json"
    save_cohort(cohort, path)
    load_cohort(path)  # untampered file loads

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["roles"] = {"source_train": [10, 20], "target_pool": []}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check"):
        load_cohort(path)


def test_blueprint_cohort_is_disjoint_and_complete():
    all_ids = [sid for ids in BLUEPRINT_COHORT.values() for sid in ids]
    assert len(all_ids) == 55
    assert len(set(all_ids)) == 55
    assert not (set(all_ids) & HISTORICAL_PILOT_SLIDES)


def test_blueprint_cohort_validates_against_real_metadata(real_metadata_paths):
    if real_metadata_paths is None:
        pytest.skip("MIDOG++ metadata not downloaded")
    manifest_path, annotation_path = real_metadata_paths
    index = build_index(manifest_path, annotation_path)
    cohort = build_cohort(index, manifest_path)
    assert cohort.roles["source_train"] == BLUEPRINT_COHORT["source_train"]
    assert cohort.role_metadata["target_pool"]["n_slides"] == 12
    # All 55 specimens are official-train records of the correct tumour types.
    assert all(v["official_splits"] == {"train": v["n_slides"]} for v in cohort.role_metadata.values())
    # Melanoma is acquired on a single scanner: tumour and scanner are confounded.
    assert list(cohort.role_metadata["target_pool"]["scanners"]) == ["Hamamatsu XR"]
