"""Sliding-window inference over whole specimen images.

The blueprint's protocol uses 512x512 tiles with stride 460 (10% overlap) and suppresses
duplicates across the whole specimen by centre distance. Tiles always stay inside the
image (the final start is snapped to ``length - tile``) so no region is missed and no
detection is produced outside the image.

Raw detections are kept with their scores above a low floor so that any threshold can be
reapplied later without re-running inference; this is what makes threshold selection on
calibration specimens honest and re-evaluation cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.images import read_rgb
from ..evaluation.evaluate import SlidePrediction
from ..evaluation.nms import nms_keep
from ..models.detector import FOREGROUND_LABEL


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering ``[0, length)`` with the last tile snapped to the edge."""
    if tile <= 0 or stride <= 0:
        raise ValueError("tile and stride must be positive")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def tile_grid(height: int, width: int, tile: int, stride: int) -> list[tuple[int, int]]:
    return [(x, y) for y in tile_starts(height, tile, stride) for x in tile_starts(width, tile, stride)]


def _normalise_batch(arrays: list[np.ndarray], mean, std) -> list[torch.Tensor]:
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    out = []
    for arr in arrays:
        x = arr.astype(np.float32) / 255.0
        x = (x - mean) / std
        out.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
    return out


@torch.no_grad()
def predict_array(
    model,
    rgb: np.ndarray,
    tile: int = 512,
    stride: int = 460,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    mean=(0.5, 0.5, 0.5),
    std=(0.25, 0.25, 0.25),
    score_floor: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Run tiled inference on an RGB array; return centres and scores in array pixels."""
    height, width = rgb.shape[:2]
    coords = tile_grid(height, width, tile, stride)
    model.eval()

    all_points: list[np.ndarray] = []
    all_scores: list[float] = []

    for start in range(0, len(coords), batch_size):
        chunk = coords[start : start + batch_size]
        crops = [rgb[y : y + tile, x : x + tile] for x, y in chunk]
        tensors = _normalise_batch(crops, mean, std)
        tensors = [t.to(device) for t in tensors]
        outputs = model(tensors)
        for (x0, y0), out in zip(chunk, outputs):
            keep = (out["labels"] == FOREGROUND_LABEL) & (out["scores"] >= score_floor)
            boxes = out["boxes"][keep].detach().cpu().numpy()
            scores = out["scores"][keep].detach().cpu().numpy()
            if len(scores) == 0:
                continue
            centres = np.stack(
                [(boxes[:, 0] + boxes[:, 2]) / 2 + x0, (boxes[:, 1] + boxes[:, 3]) / 2 + y0], axis=1
            )
            all_points.append(centres)
            all_scores.extend(scores.tolist())

    if not all_points:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    return np.concatenate(all_points, axis=0), np.asarray(all_scores, dtype=float)


@torch.no_grad()
def predict_slide(
    model,
    image_path: str | Path,
    tile: int = 512,
    stride: int = 460,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    mean=(0.5, 0.5, 0.5),
    std=(0.25, 0.25, 0.25),
    score_floor: float = 0.01,
    nms_radius_px: float | None = 25.0,
) -> SlidePrediction:
    """Predict on one specimen file.

    ``nms_radius_px=None`` keeps every raw detection (useful before threshold selection);
    otherwise whole-image centre-distance suppression is applied immediately.
    """
    rgb = read_rgb(image_path)
    slide_id = int(Path(image_path).stem)
    points, scores = predict_array(
        model,
        rgb,
        tile=tile,
        stride=stride,
        batch_size=batch_size,
        device=device,
        mean=mean,
        std=std,
        score_floor=score_floor,
    )
    if nms_radius_px is not None and len(scores):
        points, scores = nms_keep(points, scores, radius=nms_radius_px, threshold=None)
    return SlidePrediction(slide_id=slide_id, points=points, scores=scores)


@torch.no_grad()
def predict_regions(
    model,
    image_path: str | Path,
    regions,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    mean=(0.5, 0.5, 0.5),
    std=(0.25, 0.25, 0.25),
    score_floor: float = 0.01,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Detect inside a list of non-overlapping regions of one specimen.

    Regions are already fully supervised 512x512 crops, so no tiling is needed. Output
    coordinates are absolute image pixels, which lets the caller compare predicted counts
    with region ground truth without a coordinate mix-up.
    """
    rgb = read_rgb(image_path)
    model.eval()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for start in range(0, len(regions), batch_size):
        chunk = regions[start : start + batch_size]
        crops = [rgb[r.y : r.y + r.height, r.x : r.x + r.width] for r in chunk]
        tensors = [t.to(device) for t in _normalise_batch(crops, mean, std)]
        outputs = model(tensors)
        for region, pred in zip(chunk, outputs):
            keep = (pred["labels"] == FOREGROUND_LABEL) & (pred["scores"] >= score_floor)
            boxes = pred["boxes"][keep].detach().cpu().numpy()
            scores = pred["scores"][keep].detach().cpu().numpy()
            if len(scores) == 0:
                out[region.region_id] = (np.zeros((0, 2)), np.zeros(0))
                continue
            centres = np.stack(
                [
                    (boxes[:, 0] + boxes[:, 2]) / 2 + region.x,
                    (boxes[:, 1] + boxes[:, 3]) / 2 + region.y,
                ],
                axis=1,
            )
            out[region.region_id] = (centres, scores)
    return out
