"""RetinaNet detector construction, checkpointing and strict state-dict auditing.

Label convention (verified against the installed torchvision 0.22 source, not assumed):

* ``num_classes`` **includes background**. The classification head emits
  ``num_anchors * num_classes`` logits and the focal loss builds a multi-hot target by
  setting ``gt_classes_target[anchor, label] = 1`` for foreground anchors. Background is
  the implicit all-zero row, so the single mitotic-figure class must be label **1**.
* Consequence: MIDOG++ category 1 (``mitotic figure``) maps to label 1, and category 2
  (``not mitotic figure``) is **never** used as a foreground box. The historical pilot's
  category-1/category-2 confusion is avoided by this mapping being defined in one place.
* Post-processing also emits label indices modulo ``num_classes``, so detections are
  filtered with ``labels == 1``; this matches the official evaluator, which keeps only
  ``cls == 1``.

COCO weights are *initialisation*, not a MIDOG++ pretrained detector. The classification
head cannot be reused across class counts, so it is re-initialised and that fact is
verified rather than hidden behind ``strict=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..utils import load_json, save_json

NUM_CLASSES = 2  # background (implicit) + mitotic figure
FOREGROUND_LABEL = 1
#: Prefixes that legitimately differ when swapping the COCO 91-class head for ours.
HEAD_PREFIXES = ("head.classification_head.cls_logits",)


@dataclass
class DetectorConfig:
    arch: str = "retinanet_resnet50_fpn_v2"
    num_classes: int = NUM_CLASSES
    foreground_label: int = FOREGROUND_LABEL
    pretrained: str = "coco"  # 'coco' | 'imagenet_backbone' | 'none'
    trainable_backbone_layers: int = 3
    min_size: int = 512
    max_size: int = 512
    score_thresh: float = 0.01  # low floor: thresholds are applied offline

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "DetectorConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


def _check_arch(arch: str) -> None:
    if arch != "retinanet_resnet50_fpn_v2":
        raise ValueError(
            f"unsupported architecture {arch!r}; this project uses "
            "retinanet_resnet50_fpn_v2 (ResNet-50-FPN-v2) as in the research blueprint"
        )


def load_compatible_state_dict(
    model: nn.Module, state_dict: dict, allowed_missing_prefixes: tuple[str, ...] = HEAD_PREFIXES
) -> dict:
    """Load pretrained weights, skipping only shape-mismatched tensors we expect to skip.

    ``load_state_dict(strict=False)`` still raises on shape mismatches, which is exactly
    what happens to the classification head when COCO's 91 classes become our 2. So the
    tensors are filtered by shape first, and then the result is audited: anything skipped
    must match ``allowed_missing_prefixes``, and the source must not contain keys the model
    has never heard of. This keeps a silently uninitialised head from masquerading as a
    pretrained detector.
    """
    model_state = model.state_dict()
    compatible, skipped = {}, []
    for key, tensor in state_dict.items():
        target = model_state.get(key)
        if target is not None and tuple(target.shape) == tuple(tensor.shape):
            compatible[key] = tensor
        else:
            skipped.append(key)

    bad_skipped = [k for k in skipped if not k.startswith(allowed_missing_prefixes)]
    if bad_skipped:
        raise RuntimeError(f"unexpected shape mismatches: {bad_skipped}")

    unknown = [k for k in state_dict if k not in model_state]
    if unknown:
        raise RuntimeError(f"checkpoint contains keys the model does not use: {unknown}")

    load_into = model.load_state_dict(compatible, strict=False)
    return {
        "missing_keys": list(load_into.missing_keys),
        "unexpected_keys": list(load_into.unexpected_keys),
        "reinitialised": skipped,
        "n_loaded_tensors": len(compatible),
    }


def build_detector(cfg: DetectorConfig | dict | None = None) -> nn.Module:
    """Construct the detector with the project's fixed class convention."""
    cfg = cfg if isinstance(cfg, DetectorConfig) else DetectorConfig.from_dict(cfg)
    _check_arch(cfg.arch)
    if cfg.num_classes != NUM_CLASSES:
        raise ValueError(
            f"num_classes must be {NUM_CLASSES} (background + mitotic figure); got {cfg.num_classes}"
        )

    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights
    from torchvision.models import ResNet50_Weights

    kwargs = dict(
        num_classes=cfg.num_classes,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        trainable_backbone_layers=cfg.trainable_backbone_layers,
        score_thresh=cfg.score_thresh,
    )

    if cfg.pretrained == "coco":
        # Build with no weights, then load COCO detectron weights except the class head.
        model = retinanet_resnet50_fpn_v2(
            weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V2, **kwargs
        )
        state = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=False)
        model._load_report = load_compatible_state_dict(model, state)  # type: ignore[attr-defined]
    elif cfg.pretrained == "imagenet_backbone":
        model = retinanet_resnet50_fpn_v2(
            weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V2, **kwargs
        )
        model._load_report = {"missing_keys": [], "unexpected_keys": [], "reinitialised": ["all heads"]}  # type: ignore[attr-defined]
    elif cfg.pretrained == "none":
        model = retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None, **kwargs)
        model._load_report = {"missing_keys": [], "unexpected_keys": [], "reinitialised": ["all"]}  # type: ignore[attr-defined]
    else:
        raise ValueError(f"unknown pretrained setting {cfg.pretrained!r}")

    return model


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    detector_cfg: DetectorConfig | dict,
    normalization: dict,
    meta: dict | None = None,
) -> Path:
    """Save weights plus everything needed to reload them unambiguously."""
    cfg = detector_cfg if isinstance(detector_cfg, DetectorConfig) else DetectorConfig.from_dict(detector_cfg)
    payload = {
        "model_state": model.state_dict(),
        "detector_config": cfg.__dict__,
        "normalization": {
            "mean": list(normalization.get("mean", [0.0, 0.0, 0.0])),
            "std": list(normalization.get("std", [1.0, 1.0, 1.0])),
        },
        "meta": meta or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def read_checkpoint_meta(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "detector_config": payload.get("detector_config", {}),
        "normalization": payload.get("normalization", {}),
        "meta": payload.get("meta", {}),
    }


def set_trainable_backbone_layers(model: nn.Module, trainable_layers: int) -> list[str]:
    """Reproduce torchvision's backbone-freezing rule on an already-built model.

    ``retinanet_resnet50_fpn_v2`` only freezes the stem and early stages when pretrained
    weights are requested. Constructing the same architecture with ``pretrained="none"``
    leaves every layer trainable, which would give a reloaded checkpoint a different
    optimisable parameter set than the model that trained it. This applies the published
    rule:

        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:n]
        (plus "bn1" when n == 5)

    Returns the names that remain trainable, so callers can assert parity.
    """
    if not 0 <= trainable_layers <= 5:
        raise ValueError(f"trainable_backbone_layers must be within 0..5, got {trainable_layers}")
    body = model.backbone.body
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in body.named_parameters():
        parameter.requires_grad_(any(name.startswith(layer) for layer in layers_to_train))
    return sorted(
        f"backbone.body.{name}" for name, p in body.named_parameters() if p.requires_grad
    )


def require_device(device: str | torch.device) -> str:
    """Reject a cuda device on a CPU-only torch build with one readable line.

    Without this, ``model.to("cuda")`` fails deep inside ``nn.Module._apply`` with
    ``AssertionError: Torch not compiled with CUDA enabled``, after a run directory has
    already been created. The device actually available is what the caller needs to know.
    """
    name = str(device)
    if name.split(":")[0] != "cpu" and not torch.cuda.is_available():
        raise SystemExit(
            f"device '{name}' requested but this torch build has no CUDA "
            f"(torch {torch.__version__}, cuda available: False). "
            f"Use device 'cpu', or run on a CUDA build/runtime."
        )
    return name


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the architecture from the checkpoint's own config and load its weights.

    The architecture is constructed **without** pretrained weights: every tensor is
    overwritten by the checkpoint, so fetching ImageNet/COCO weights would only cost a
    ~150 MB download per reload (and fail outright on an offline runtime). The backbone
    freezing that the pretrained path would have applied is reproduced explicitly, so the
    reloaded model has exactly the same trainable parameter set as the one that produced
    the checkpoint.

    The classification head shape is checked explicitly so a checkpoint trained with a
    different class convention cannot be silently mis-loaded.
    """
    from dataclasses import replace

    require_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = DetectorConfig.from_dict(payload.get("detector_config") or {})
    model = build_detector(replace(cfg, pretrained="none"))
    if cfg.pretrained != "none":
        model._trainable_backbone_names = set_trainable_backbone_layers(  # type: ignore[attr-defined]
            model, cfg.trainable_backbone_layers
        )

    state = payload["model_state"]
    head_key = "head.classification_head.cls_logits.weight"
    if head_key in state:
        expected = cfg.num_classes
        anchors = 9  # RetinaNet defaults: 5 scales x 3 aspect ratios
        actual = int(state[head_key].shape[0])
        if actual != expected * anchors:
            raise RuntimeError(
                f"checkpoint head has {actual} outputs, incompatible with "
                f"num_classes={expected} x {anchors} anchors"
            )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, payload


def describe_checkpoint(path: str | Path) -> dict:
    """Human-readable provenance summary for reports."""
    meta = read_checkpoint_meta(path)
    return {
        "path": str(path),
        "detector_config": meta["detector_config"],
        "normalization": meta["normalization"],
        "meta": meta["meta"],
    }


def export_provenance(path: str | Path, out_path: str | Path) -> Path:
    return save_json(describe_checkpoint(path), out_path)
