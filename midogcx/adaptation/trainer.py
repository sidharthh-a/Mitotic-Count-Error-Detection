"""Detector training and controlled target adaptation.

Every adaptation branch starts from the **same** source checkpoint, trains for the same
number of updates with the same optimiser, schedule, augmentation, sampling ratio and
evaluation protocol. Only the selected target regions differ. That is what makes the
strategy comparison meaningful; without it a difference could come from any of those
knobs.

Training is CPU-safe (it will simply be slow) and CUDA-ready, so the same entry point
runs locally for smoke tests and on Colab for real experiments.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import (
    MitosisRegionDataset,
    TrainingItem,
    detection_collate,
    positive_enriched_indices,
)
from ..models.detector import DetectorConfig, save_checkpoint
from ..utils import get_logger, seed_everything


@dataclass
class TrainConfig:
    updates: int = 500
    batch_size: int = 2
    lr_head: float = 5e-4
    lr_backbone: float = 5e-5
    weight_decay: float = 1e-4
    warmup_updates: int = 50
    samples_per_epoch: int = 64
    fraction_positive: float = 0.5
    grad_clip: float = 10.0
    log_every: int = 20
    checkpoint_every: int = 0  # 0 = only at the end
    num_workers: int = 0
    replay_probability: float = 0.0  # share of batches drawn from the source domain
    device: str = "cpu"
    seed: int = 0
    deterministic: bool = True

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "TrainConfig":
        cfg = cfg or {}
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _lr_lambda(update: int, warmup: int, total: int) -> float:
    if warmup > 0 and update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress))))


def build_optimizer(model, cfg: TrainConfig):
    """Discriminative learning rates: pretrained backbone moves slower than the head."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if name.startswith("backbone") else head_params).append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    groups.append({"params": head_params, "lr": cfg.lr_head})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def mixed_indices(
    target_items: list[TrainingItem],
    source_items: list[TrainingItem],
    n_samples: int,
    fraction_positive: float,
    replay_probability: float,
    rng: np.random.Generator,
) -> list[int]:
    """Index list into ``target_items + source_items`` for one epoch.

    ``replay_probability`` is the fraction of draws taken from the source domain; 0 means
    target-only adaptation, as in the bounded feasibility experiment.
    """
    n_replay = int(round(n_samples * replay_probability)) if source_items else 0
    n_target = n_samples - n_replay
    out = positive_enriched_indices(target_items, n_target, fraction_positive, rng) if target_items else []
    if n_replay:
        offset = len(target_items)
        out += [offset + i for i in positive_enriched_indices(source_items, n_replay, fraction_positive, rng)]
    rng.shuffle(out)
    return out


def train_detector(
    model,
    target_items: list[TrainingItem],
    index,
    data_root,
    train_cfg: TrainConfig,
    run_dir: str | Path,
    detector_cfg: DetectorConfig | dict,
    normalization: dict,
    source_items: list[TrainingItem] | None = None,
    meta: dict | None = None,
) -> dict:
    """Fine-tune ``model`` for ``updates`` steps and save a checkpoint plus a log.

    Returns a summary dict with the loss trace, timing and checkpoint path. The loss trace
    is what a future reader needs to see that training actually happened; it is not an
    accuracy estimate.
    """
    train_cfg = train_cfg if isinstance(train_cfg, TrainConfig) else TrainConfig.from_dict(train_cfg)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"train.{run_dir.name}", run_dir / "train.log")

    seed_everything(train_cfg.seed, deterministic_torch=train_cfg.deterministic)
    rng = np.random.default_rng(train_cfg.seed)
    source_items = source_items or []

    combined = list(target_items) + list(source_items)
    if not combined:
        raise ValueError("no training items supplied")

    dataset = MitosisRegionDataset(
        combined,
        index=index,
        data_root=data_root,
        mean=normalization["mean"],
        std=normalization["std"],
        augment_data=True,
        seed=train_cfg.seed,
    )
    device = torch.device(train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu")
    model.to(device)
    model.train()

    optimizer = build_optimizer(model, train_cfg)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    trace: list[dict] = []
    update = 0
    start = time.time()
    logger.info(
        "training on %d target + %d source regions, %d updates, device=%s",
        len(target_items),
        len(source_items),
        train_cfg.updates,
        device,
    )

    while update < train_cfg.updates:
        indices = mixed_indices(
            target_items,
            source_items,
            train_cfg.samples_per_epoch,
            train_cfg.fraction_positive,
            train_cfg.replay_probability,
            rng,
        )
        loader = DataLoader(
            torch.utils.data.Subset(dataset, indices),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            collate_fn=detection_collate,
            drop_last=False,
        )
        for images, targets in loader:
            if update >= train_cfg.updates:
                break
            scale = _lr_lambda(update, train_cfg.warmup_updates, train_cfg.updates)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * scale

            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], train_cfg.grad_clip
            )
            optimizer.step()
            update += 1

            if update % train_cfg.log_every == 0 or update == 1:
                record = {
                    "update": update,
                    "loss": float(loss.detach().cpu()),
                    "classification": float(loss_dict["classification"].detach().cpu()),
                    "bbox_regression": float(loss_dict["bbox_regression"].detach().cpu()),
                    "lr_head": optimizer.param_groups[-1]["lr"],
                    "elapsed_s": round(time.time() - start, 1),
                }
                trace.append(record)
                logger.info(
                    "update %d/%d loss %.4f (cls %.4f, box %.4f) %.1fs",
                    update,
                    train_cfg.updates,
                    record["loss"],
                    record["classification"],
                    record["bbox_regression"],
                    record["elapsed_s"],
                )

            if train_cfg.checkpoint_every and update % train_cfg.checkpoint_every == 0:
                save_checkpoint(
                    run_dir / f"checkpoint_u{update}.pth",
                    model,
                    detector_cfg,
                    normalization,
                    meta={**(meta or {}), "updates": update, "partial": True},
                )

    elapsed = time.time() - start
    ckpt_path = save_checkpoint(
        run_dir / "checkpoint.pth",
        model,
        detector_cfg,
        normalization,
        meta={**(meta or {}), "updates": update, "elapsed_s": round(elapsed, 1)},
    )

    summary = {
        "updates": update,
        "elapsed_s": round(elapsed, 1),
        "seconds_per_update": round(elapsed / max(1, update), 3),
        "n_target_regions": len(target_items),
        "n_source_replay_regions": len(source_items),
        "train_config": train_cfg.__dict__,
        "checkpoint": str(ckpt_path),
        "loss_trace": trace,
        "final_loss": trace[-1]["loss"] if trace else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model.eval()
    return summary


# --------------------------------------------------------------------------------------
# Region loss (supervision for the learned-loss baseline)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def detector_loss_per_region(
    model,
    items: list[TrainingItem],
    index,
    data_root,
    normalization: dict,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Detector training loss for each *labelled* region.

    This is the supervision for the generic predicted-loss acquisition rule. It uses the
    same model, the same normalisation and the same targets as count-error prediction, so
    the two learners differ only in what they are asked to predict.

    Regions are scored one at a time because torchvision's detection losses are already
    normalised per image, so a batch mean cannot be attributed back to individual regions.

    Detection losses only exist in ``train()`` mode, so the model is switched to train mode
    for the forward passes. That would otherwise **overwrite every BatchNorm running
    statistic** with single-image statistics from source-domain crops, silently changing
    the detector before it is used for pool features and for the ``source_only`` control
    (which is evaluated after this call). All buffers and the original train/eval mode are
    therefore snapshotted and restored, so this function has no side effects.

    Known consequence: because BatchNorm uses the batch's own statistics here rather than
    its running estimates, the scalar loss is computed under a slightly different
    normalisation regime than inference. This is inherent to torchvision's loss API and is
    recorded in docs/EXPERIMENTAL_READINESS_AUDIT.md rather than hidden.
    """
    dataset = MitosisRegionDataset(
        items,
        index=index,
        data_root=data_root,
        mean=normalization["mean"],
        std=normalization["std"],
        augment_data=False,
    )
    model.to(device)

    buffers_before = {name: buf.detach().clone() for name, buf in model.named_buffers()}
    was_training = model.training
    model.train()  # detection losses are only defined in train mode
    out: dict[str, float] = {}
    try:
        for i, item in enumerate(items):
            image, target = dataset[i]
            loss_dict = model([image.to(device)], [{k: v.to(device) for k, v in target.items()}])
            out[item.region_id] = float(loss_dict["classification"].detach().cpu()) + float(
                loss_dict["bbox_regression"].detach().cpu()
            )
    finally:
        with torch.no_grad():
            for name, buf in model.named_buffers():
                buf.copy_(buffers_before[name])
        model.train(was_training)
    return out
