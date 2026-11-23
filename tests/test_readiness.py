"""Integration tests for the experiment's scientific invariants.

These are the checks that must hold before any GPU time is spent:

* ranking must not mutate the detector it is ranking with;
* out-of-fold grouping must stay aligned with the feature matrix;
* every strategy must rank the same pool under the same physical-area budget;
* the density baseline must count the same deduplicated detections as everything else;
* a reloaded checkpoint must be the same object the source run trained, including the
  pretrained backbone-freezing pattern, without re-fetching pretrained weights.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from midogcx.adaptation.trainer import detector_loss_per_region
from midogcx.data.patches import Region
from midogcx.models.detector import (
    DetectorConfig,
    build_detector,
    load_checkpoint,
    save_checkpoint,
)
from midogcx.selection.strategies import StrategyContext, rank_regions

REPO = Path(__file__).resolve().parents[1]


def _experiment_module():
    spec = importlib.util.spec_from_file_location(
        "run_acquisition_experiment", REPO / "scripts" / "run_acquisition_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _region(rid: str, sid: int, area_um2: float, n_pos: int = 0, role: str = "target_pool"):
    return Region(rid, sid, 0, 0, 512, 512, int(area_um2 / 0.0625), area_um2, n_pos, role)


# --------------------------------------------------------------------------------------
# 1. Ranking must not mutate the detector (BatchNorm statistics)
# --------------------------------------------------------------------------------------
def test_region_loss_computation_preserves_buffers_and_mode(monkeypatch):
    import midogcx.adaptation.trainer as trainer

    model = build_detector(DetectorConfig(pretrained="none"))
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    class _FakeDataset:
        def __init__(self, items, **kwargs):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            image = torch.full((3, 512, 512), 0.5)
            target = {
                "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
                "labels": torch.tensor([1]),
                "area": torch.tensor([900.0]),
                "iscrowd": torch.tensor([0]),
            }
            return image, target

    monkeypatch.setattr(trainer, "MitosisRegionDataset", _FakeDataset)

    class _Item:
        def __init__(self, rid: str) -> None:
            self.region_id = rid

    items = [_Item("001_0_0"), _Item("001_512_0")]
    losses = detector_loss_per_region(
        model,
        items,
        index=None,
        data_root=".",
        normalization={"mean": [0.5] * 3, "std": [0.25] * 3},
        device="cpu",
    )

    assert len(losses) == 2
    after = model.state_dict()
    changed = [k for k, v in before.items() if not torch.equal(v, after[k])]
    assert changed == [], f"loss computation mutated {len(changed)} tensors: {changed[:5]}"
    assert model.training is False, "model mode was not restored"


# --------------------------------------------------------------------------------------
# 2 & 3. Out-of-fold groups and deduplicated counts through the real experiment helper
# --------------------------------------------------------------------------------------
def test_region_features_groups_align_and_counts_are_deduplicated(monkeypatch):
    exp = _experiment_module()
    log = __import__("logging").getLogger("readiness")
    log.addHandler(__import__("logging").NullHandler())

    regions = [
        _region("001_0_0", 1, 1_000_000.0, role="source_calib"),
        _region("002_0_0", 2, 1_000_000.0, role="source_calib"),
        _region("001_512_0", 1, 1_000_000.0, role="source_calib"),
    ]
    # Each region gets three detections, two of them within the 25 px NMS radius.
    monkeypatch.setattr(
        exp,
        "predict_regions",
        lambda model, path, regs, **kw: {
            r.region_id: (
                np.array([[100.0, 100.0], [105.0, 100.0], [400.0, 400.0]]),
                np.array([0.9, 0.8, 0.7]),
            )
            for r in regs
        },
    )
    monkeypatch.setattr(exp, "resolve_specimen_path", lambda root, name: Path(name))

    features, counts, detections, groups = exp._region_features(
        None,
        regions,
        ".",
        {"batch_size": 1, "score_floor": 0.01, "nms_radius_px": 25},
        {"mean": [0.5] * 3, "std": [0.25] * 3},
        "cpu",
        0.5,
        log,
    )

    expected_groups = [r.slide_id for r in regions]
    assert list(groups) == expected_groups, (
        "group vector must follow the region list order, not inference order; "
        f"got {list(groups)} expected {expected_groups}"
    )
    assert set(features) == {r.region_id for r in regions}

    # Two of three detections survive suppression, so every consumer must see 2.
    for rid in counts:
        assert counts[rid] == 2, f"{rid} count {counts[rid]} was not deduplicated"
        kept_points, kept_scores = detections[rid]
        assert len(kept_scores) == 2
        assert int((kept_scores > 0.5).sum()) == counts[rid]


def test_density_uses_same_counts_as_locked_counting():
    """The density baseline must not count duplicates that NMS removed elsewhere."""
    exp = _experiment_module()
    log = __import__("logging").getLogger("readiness2")
    log.addHandler(__import__("logging").NullHandler())

    region = _region("001_0_0", 1, 1_000_000.0)
    exp.predict_regions = lambda model, path, regs, **kw: {
        r.region_id: (
            np.array([[100.0, 100.0], [105.0, 100.0], [400.0, 400.0]]),
            np.array([0.9, 0.8, 0.7]),
        )
        for r in regs
    }
    exp.resolve_specimen_path = lambda root, name: Path(name)

    features, counts, detections, _ = exp._region_features(
        None, [region], ".", {"batch_size": 1, "score_floor": 0.01, "nms_radius_px": 25},
        {"mean": [0.5] * 3, "std": [0.25] * 3}, "cpu", 0.5, log,
    )
    ctx = StrategyContext(detections=detections, threshold=0.5, seed=0, features=features)
    density = dict(rank_regions([region], "density", ctx))
    # 1 mm^2 of tissue, so the density value is the count in figures per mm^2.
    assert density[region.region_id] == pytest.approx(float(counts[region.region_id]))


def test_nms_is_idempotent_so_filter_then_suppress_equals_suppress_then_filter():
    from midogcx.evaluation.nms import nms_keep

    rng = np.random.default_rng(0)
    points = rng.uniform(0, 200, size=(40, 2))
    scores = rng.uniform(0.2, 1.0, size=40)

    once = nms_keep(points, scores, radius=25.0, threshold=None)
    twice = nms_keep(once[0], once[1], radius=25.0, threshold=None)
    assert len(once[1]) == len(twice[1])

    suppress_first = len(nms_keep(points, scores, radius=25.0, threshold=None)[1])
    for thr in (0.3, 0.5, 0.7):
        a = nms_keep(points, scores, radius=25.0, threshold=thr)
        kept = nms_keep(points, scores, radius=25.0, threshold=None)
        b_mask = kept[1] > thr
        assert len(a[1]) == int(b_mask.sum())


# --------------------------------------------------------------------------------------
# 4. Pool identity and equal physical-area budget across strategies
# --------------------------------------------------------------------------------------
class _StubPredictor:
    feature_names = ("density_feature",)

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


def test_disjoint_guard_rejects_overlapping_regions():
    from midogcx.selection.budget import assert_disjoint_regions

    a = Region("001_0_0", 1, 0, 0, 512, 512, 1000, 1000.0, 0, "target_pool")
    b = Region("001_256_0", 1, 256, 0, 512, 512, 1000, 1000.0, 0, "target_pool")
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_regions([a, b])

    # Same coordinates on a different specimen is not an overlap.
    other_slide = Region("002_256_0", 2, 256, 0, 512, 512, 1000, 1000.0, 0, "target_pool")
    assert_disjoint_regions([a, other_slide])


def test_disjoint_guard_accepts_a_real_grid():
    from midogcx.selection.budget import assert_disjoint_regions

    grid = [
        Region(f"001_{x}_{y}", 1, x, y, 512, 512, 1000, 1000.0, 0, "target_pool")
        for y in (0, 512, 1024)
        for x in (0, 512)
    ]
    assert_disjoint_regions(grid)


def test_all_strategies_share_pool_and_budget():
    rng = np.random.default_rng(3)
    pool = [
        _region(f"100_{i * 512}_0", 100, float(rng.uniform(0.4e6, 2.0e6))) for i in range(25)
    ]
    detections = {}
    features = {}
    for r in pool:
        n = int(rng.integers(0, 5))
        pts = rng.uniform(0, 512, size=(n, 2))
        scs = rng.uniform(0.3, 0.99, size=n)
        detections[r.region_id] = (pts, scs)
        features[r.region_id] = {"density_feature": float(n) + float(rng.uniform(0, 0.1))}

    ctx = StrategyContext(
        detections=detections,
        threshold=0.5,
        seed=7,
        predictors={"count_error": _StubPredictor(), "learned_loss": _StubPredictor()},
        features=features,
    )
    from midogcx.selection.budget import budget_from_fraction, select_under_budget

    from midogcx.selection.strategies import STRATEGIES

    lookup = {r.region_id: r for r in pool}
    pool_ids = set(lookup)
    areas = {}
    for name in STRATEGIES:
        ranking = rank_regions(pool, name, ctx)
        assert {rid for rid, _ in ranking} == pool_ids, f"{name} ranked a different pool"
        assert len(ranking) == len(pool_ids), f"{name} ranked regions twice"
        budget = budget_from_fraction(pool, 0.2)
        chosen = select_under_budget(lookup, ranking, budget)
        assert len({r.region_id for r in chosen}) == len(chosen)
        assert budget.used_um2 <= budget.total_um2 + 1e-6
        areas[name] = budget.used_um2

    largest = max(r.tissue_area_um2 for r in pool)
    spread = max(areas.values()) - min(areas.values())
    assert spread <= largest + 1e-6, f"budgets differ by {spread} um2 > one region ({largest})"


# --------------------------------------------------------------------------------------
# 5 & 6. Checkpoint reload is faithful and free of pretrained downloads
# --------------------------------------------------------------------------------------
def test_load_checkpoint_does_not_fetch_pretrained_and_keeps_freeze_pattern(tmp_path):
    cfg = DetectorConfig(pretrained="coco", trainable_backbone_layers=3)
    model = build_detector(DetectorConfig(pretrained="none"))
    from midogcx.models.detector import set_trainable_backbone_layers

    set_trainable_backbone_layers(model, 3)
    path = save_checkpoint(
        tmp_path / "ckpt.pth", model, cfg, {"mean": [0.5] * 3, "std": [0.25] * 3}, meta={}
    )

    loaded, payload = load_checkpoint(path, device="cpu")
    assert payload["detector_config"]["pretrained"] == "coco"

    # Built through the uninitialised path, so nothing was downloaded to be overwritten.
    assert loaded._load_report["reinitialised"] == ["all"]

    for name, param in loaded.named_parameters():
        original = dict(model.named_parameters())[name]
        assert param.requires_grad == original.requires_grad, f"{name} trainability changed"
        assert torch.equal(param, original), f"{name} weights differ after reload"

    trainable = {n for n, p in loaded.named_parameters() if p.requires_grad}
    assert not any(n.startswith("backbone.body.layer1.") for n in trainable)
    assert any(n.startswith("backbone.body.layer3.") for n in trainable)
    assert any(n.startswith("head.") for n in trainable)


def test_threshold_tie_break_prefers_higher_threshold():
    from midogcx.evaluation.evaluate import SlidePrediction, select_threshold

    preds = {1: SlidePrediction(1, np.array([[10.0, 10.0], [900.0, 900.0]]), np.array([0.75, 0.75]))}
    info = select_threshold(
        preds,
        [1],
        gt_fn=lambda sid: np.array([[10.0, 10.0]]),
        spacing={1: (0.25, 0.25)},
        thresholds=np.round(np.arange(0.5, 0.99, 0.01), 2),
        nms_radius_px=25.0,
        match_radius_um=7.5,
    )
    tying = [row["threshold"] for row in info["sweep"] if row["f1"] == pytest.approx(info["sweep"][0]["f1"])]
    assert info["best_threshold"] == pytest.approx(max(tying))
