"""Torch dataset for detector training on tissue regions.

Label rules that matter for scientific validity:

* Only category-1 (mitotic figure) annotations become foreground boxes, with label **1**
  (torchvision RetinaNet reserves index 0 for background). Category-2 boxes are curated
  *non-mitotic* cells: they are not foreground and are not converted into "background
  boxes"; they simply contribute no positive target.
* A region's targets are **all** positive annotations whose centre lies inside it, not
  only the cell whose presence caused the region to be sampled. Otherwise crops would be
  trained with incomplete labels.
* Boxes are clipped to the crop only for loss computation. Evaluation never uses clipped
  boxes; it matches point centres from the original annotations.

Because only selected regions are annotated by construction, unselected tissue is never
treated as confirmed background: a training item is always a fully supervised region.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import resolve_specimen_path
from .images import read_rgb
from .metadata import MidogIndex
from .patches import Region


@dataclass(frozen=True)
class TrainingItem:
    region_id: str
    slide_id: int
    x: int
    y: int
    width: int
    height: int
    has_positive: bool
    n_positives: int = 0
    role: str = "unknown"


def items_from_regions(regions: list[Region]) -> list[TrainingItem]:
    return [
        TrainingItem(
            region_id=r.region_id,
            slide_id=r.slide_id,
            x=r.x,
            y=r.y,
            width=r.width,
            height=r.height,
            has_positive=r.n_positives > 0,
            n_positives=r.n_positives,
            role=r.role,
        )
        for r in regions
    ]


# --------------------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------------------
def _clip_boxes(boxes: np.ndarray, width: int, height: int, min_size: float = 2.0) -> np.ndarray:
    """Clip boxes to the crop and drop degenerate ones."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height)
    keep = (out[:, 2] - out[:, 0] >= min_size) & (out[:, 3] - out[:, 1] >= min_size)
    return out[keep]


def transform_boxes(boxes: np.ndarray, width: int, height: int, hflip: bool, vflip: bool, rot90: int):
    """Apply a flip/rotation to xyxy boxes and return new boxes plus new (width, height)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    w, h = float(width), float(height)
    rot90 = int(rot90) % 4

    for _ in range(rot90):
        # Counter-clockwise: (x, y) -> (y, w - x), and the canvas becomes h x w.
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes = np.stack([y1, w - x2, y2, w - x1], axis=1)
        w, h = h, w

    if hflip:
        x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
        boxes[:, 0], boxes[:, 2] = w - x2, w - x1
    if vflip:
        y1, y2 = boxes[:, 1].copy(), boxes[:, 3].copy()
        boxes[:, 1], boxes[:, 3] = h - y2, h - y1

    return boxes.astype(np.float32), int(w), int(h)


def augment(
    image: np.ndarray,
    boxes: np.ndarray,
    rng: np.random.Generator,
    enable_rotation: bool = True,
    enable_colour: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip/rotate/colour augmentation with boxes transformed consistently."""
    h, w = image.shape[:2]
    hflip = bool(rng.random() < 0.5)
    vflip = bool(rng.random() < 0.5)
    rot = int(rng.integers(0, 4)) if enable_rotation else 0

    boxes, new_w, new_h = transform_boxes(boxes, w, h, hflip, vflip, rot)
    if rot:
        image = np.rot90(image, k=rot)
    if hflip:
        image = image[:, ::-1]
    if vflip:
        image = image[::-1, :]
    image = np.ascontiguousarray(image)

    if enable_colour:
        img = image.astype(np.float32)
        if rng.random() < 0.8:
            img *= float(rng.uniform(0.85, 1.15))  # brightness
        if rng.random() < 0.5:
            mean = img.mean()
            img = (img - mean) * float(rng.uniform(0.9, 1.1)) + mean  # contrast
        image = np.clip(img, 0, 255).astype(np.uint8)

    return image, _clip_boxes(boxes, new_w, new_h)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class MitosisRegionDataset(Dataset):
    """Fully supervised 512x512 tissue regions for detector training/fine-tuning."""

    def __init__(
        self,
        items: list[TrainingItem],
        index: MidogIndex,
        data_root: str | Path,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.25, 0.25, 0.25),
        augment_data: bool = True,
        seed: int = 0,
        enable_rotation: bool = True,
    ):
        if not items:
            raise ValueError("dataset needs at least one training item")
        self.items = items
        self.index = index
        self.data_root = Path(data_root)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.augment_data = augment_data
        self.seed = int(seed)
        self.enable_rotation = enable_rotation

    def __len__(self) -> int:
        return len(self.items)

    def _read(self, item: TrainingItem) -> tuple[np.ndarray, np.ndarray]:
        path = resolve_specimen_path(self.data_root, f"{item.slide_id:03d}.tiff")
        if not path.exists():
            raise FileNotFoundError(f"specimen image missing: {path}")
        region = (item.x, item.y, item.width, item.height)
        image = read_rgb(path, region=region)
        labels = self.index.annotations_in_box(
            item.slide_id, item.x, item.y, item.width, item.height
        )
        boxes = labels["positive_boxes"].astype(np.float32).reshape(-1, 4)
        if len(boxes):
            boxes = boxes - np.array([item.x, item.y, item.x, item.y], dtype=np.float32)
            boxes = _clip_boxes(boxes, image.shape[1], image.shape[0])
        return image, boxes

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image, boxes = self._read(item)

        if self.augment_data:
            rng = np.random.default_rng(self.seed * 100003 + idx)
            image, boxes = augment(image, boxes, rng, enable_rotation=self.enable_rotation)

        arr = image.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

        target = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "area": torch.from_numpy(
                ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).astype(np.float32)
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return tensor, target


def detection_collate(batch):
    """RetinaNet expects a list of images and a list of target dicts."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def positive_enriched_indices(
    items: list[TrainingItem],
    n_samples: int,
    fraction_positive: float,
    rng: np.random.Generator,
) -> list[int]:
    """Build one epoch's index list with a fixed positive:random ratio.

    RetinaNet needs both positive and negative crops: sampling only mitosis-rich regions
    produces a flood of false positives, sampling only random tissue produces none. The
    ratio is a documented hyperparameter so every branch of the experiment uses the same
    one.
    """
    if not 0.0 <= fraction_positive <= 1.0:
        raise ValueError("fraction_positive must be within [0, 1]")
    pos = [i for i, it in enumerate(items) if it.has_positive]
    all_idx = np.arange(len(items))
    if not pos:
        return rng.choice(all_idx, size=n_samples, replace=True).tolist()

    n_pos = int(round(n_samples * fraction_positive))
    n_rand = n_samples - n_pos
    pos_pick = rng.choice(np.asarray(pos), size=n_pos, replace=len(pos) < n_pos).tolist()
    rand_pick = rng.choice(all_idx, size=n_rand, replace=len(all_idx) < n_rand).tolist()
    out = pos_pick + rand_pick
    rng.shuffle(out)
    return [int(i) for i in out]
