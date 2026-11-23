"""Detector construction, label convention and checkpoint round-trip."""

from __future__ import annotations

import pytest
import torch

from midogcx.models.detector import (
    FOREGROUND_LABEL,
    NUM_CLASSES,
    DetectorConfig,
    build_detector,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture(scope="module")
def tiny_model():
    return build_detector(DetectorConfig(pretrained="none", min_size=256, max_size=256))


def test_class_convention_is_background_plus_one_foreground():
    # torchvision counts the background class in num_classes; label 1 is the mitosis class.
    assert NUM_CLASSES == 2
    assert FOREGROUND_LABEL == 1


def test_head_emits_num_classes_times_anchors(tiny_model):
    weight = tiny_model.head.classification_head.cls_logits.weight
    assert weight.shape[0] == NUM_CLASSES * 9


def test_wrong_num_classes_is_rejected():
    with pytest.raises(ValueError, match="num_classes must be"):
        build_detector(DetectorConfig(num_classes=1, pretrained="none"))


def test_forward_returns_detection_dict(tiny_model):
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model([torch.zeros(3, 256, 256)])
    assert set(out[0]) == {"boxes", "scores", "labels"}
    assert out[0]["boxes"].shape[1] == 4


def test_training_loss_accepts_label_one_only(tiny_model):
    tiny_model.train()
    image = torch.zeros(3, 256, 256)
    target = {
        "boxes": torch.tensor([[100.0, 100.0, 140.0, 140.0]]),
        "labels": torch.tensor([FOREGROUND_LABEL]),
        "area": torch.tensor([1600.0]),
        "iscrowd": torch.tensor([0]),
    }
    losses = tiny_model([image], [target])
    assert "classification" in losses and "bbox_regression" in losses
    assert torch.isfinite(sum(losses.values()))


def test_checkpoint_roundtrip_preserves_weights(tiny_model, tmp_path):
    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    path = tmp_path / "ckpt.pth"
    save_checkpoint(path, tiny_model, DetectorConfig(pretrained="none", min_size=256, max_size=256),
                    norm, meta={"updates": 3, "role": "source_train"})

    loaded, payload = load_checkpoint(path, device="cpu")
    assert payload["meta"]["updates"] == 3
    assert payload["normalization"]["mean"] == [0.5, 0.5, 0.5]

    for (name_a, param_a), (name_b, param_b) in zip(
        tiny_model.state_dict().items(), loaded.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


def test_checkpoint_with_wrong_head_size_is_rejected(tiny_model, tmp_path):
    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    path = tmp_path / "bad.pth"
    save_checkpoint(path, tiny_model, DetectorConfig(pretrained="none", min_size=256, max_size=256), norm)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = "head.classification_head.cls_logits.weight"
    payload["model_state"][key] = payload["model_state"][key][:9]  # pretend a 1-class head
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="incompatible"):
        load_checkpoint(path)
