"""TIFF reading, physical pixel spacing and tissue masking.

MIDOG++ specimen files are single-page RGB TIFFs of ~7000x5400 px (~145 MiB each,
~117 Mpx decoded). They are too large to hold many of in memory, so this module always
works with one specimen at a time and offers a downsampled tissue mask for the fast
region-grid pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Fallback spacing when a TIFF carries no usable resolution tag.
#:
#: 0.25 um/px is the acquisition spacing of the MIDOG++ scanners, but the value is
#: always taken from the file when available and the source is recorded, because the
#: evaluator converts distances to micrometres.
DEFAULT_SPACING_UM = 0.25

_RES_UNIT_UM = {2: 25400.0, 3: 10000.0}  # 2 = inch, 3 = centimetre


@dataclass(frozen=True)
class SlideGeometry:
    path: Path
    width: int
    height: int
    spacing_x_um: float
    spacing_y_um: float
    spacing_source: str  # 'tiff_resolution' | 'imagej_metadata' | 'default_assumption'

    @property
    def area_um2_per_px(self) -> float:
        return self.spacing_x_um * self.spacing_y_um


def _from_tiff_tags(tags: dict) -> tuple[float, float] | None:
    unit = int(tags.get(296, 1) or 1)
    if unit not in _RES_UNIT_UM:
        return None
    scale = _RES_UNIT_UM[unit]

    def _ratio(value) -> float:
        try:
            num, den = value.numerator, value.denominator
            return float(num) / float(den) if den else float(num)
        except AttributeError:
            return float(value)

    xres = tags.get(282)
    yres = tags.get(283, xres)
    if xres is None or yres is None:
        return None
    xres, yres = _ratio(xres), _ratio(yres)
    if xres <= 0 or yres <= 0:
        return None
    return scale / xres, scale / yres


def _from_imagej(description: str | None) -> tuple[float, float] | None:
    """Parse ImageJ ``unit=micron`` + ``spacing=`` metadata when present."""
    if not description or "unit=" not in description:
        return None
    fields: dict[str, str] = {}
    for line in description.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            fields[key.strip().lower()] = val.strip()
    if fields.get("unit", "").lower() not in ("micron", "um", "µm"):
        return None
    try:
        spacing = float(fields.get("spacing", "nan"))
    except ValueError:
        return None
    if not np.isfinite(spacing) or spacing <= 0:
        return None
    # ImageJ "spacing" is the z-step in the same unit, not the xy pixel size, so it is
    # only used when explicit pixel sizes are absent from the tags.
    return None


def read_geometry(path: str | Path) -> SlideGeometry:
    """Read size and physical pixel spacing from a specimen TIFF."""
    from PIL import Image

    path = Path(path)
    with Image.open(path) as img:
        width, height = img.size
        tags = dict(img.tag_v2)
        description = tags.get(270)

    spacing = _from_tiff_tags(tags)
    source = "tiff_resolution"
    if spacing is None:
        spacing = _from_imagej(description)
        source = "imagej_metadata"
    if spacing is None:
        spacing = (DEFAULT_SPACING_UM, DEFAULT_SPACING_UM)
        source = "default_assumption"

    return SlideGeometry(
        path=path,
        width=int(width),
        height=int(height),
        spacing_x_um=float(spacing[0]),
        spacing_y_um=float(spacing[1]),
        spacing_source=source,
    )


def read_rgb(path: str | Path, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Load a specimen (or a crop of it) as ``uint8`` RGB.

    ``region`` is ``(x, y, w, h)`` in pixel coordinates and is clipped to the image.
    """
    from PIL import Image

    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if region is not None:
            x, y, w, h = region
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1 = min(img.width, int(x) + int(w))
            y1 = min(img.height, int(y) + int(h))
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"region {region} lies outside image {img.size}")
            img = img.crop((x0, y0, x1, y1))
        return np.asarray(img, dtype=np.uint8)


def downsample_rgb(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample by an integer factor (keeps colour, cheap for masking)."""
    if factor <= 1:
        return rgb
    h, w = rgb.shape[:2]
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    view = rgb[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor, -1)
    return view.mean(axis=(1, 3)).astype(np.uint8)


def tissue_mask(rgb: np.ndarray, downsample: int = 8) -> tuple[np.ndarray, int]:
    """Boolean tissue mask at ``1/downsample`` resolution plus the factor used.

    Otsu thresholding on luminance separates stained tissue from the white glass
    background. Small speckles are removed with a morphological opening so that
    dust and pen marks do not create spurious "tissue area".
    """
    import cv2

    small = downsample_rgb(rgb, downsample)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    # Otsu returns the threshold that maximises between-class variance; background is
    # the bright mode, so everything darker than the threshold is tissue.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = gray < thresh
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool), downsample


def crop_tissue_fraction(mask: np.ndarray, factor: int, x: int, y: int, w: int, h: int) -> float:
    """Fraction of tissue pixels inside the pixel-space box, using a downsampled mask."""
    mx0, my0 = max(0, x // factor), max(0, y // factor)
    mx1, my1 = min(mask.shape[1], (x + w + factor - 1) // factor), min(
        mask.shape[0], (y + h + factor - 1) // factor
    )
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    patch = mask[my0:my1, mx0:mx1]
    # Approximate the pixel-space denominator from the downsampled box.
    denom = patch.size
    return float(patch.sum()) / float(denom) if denom else 0.0


def image_mean_std(paths: list[str | Path], max_side: int = 512) -> tuple[list[float], list[float]]:
    """Estimate per-channel mean/std (0-1 scale) over a set of specimens.

    Images are downsampled before averaging so this stays cheap; the result is stored in
    every checkpoint so inference can never silently use different normalisation.
    """
    from PIL import Image

    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for p in paths:
        with Image.open(p) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((max_side, max_side))
            arr = np.asarray(img, dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        sums += flat.sum(axis=0)
        sums_sq += (flat**2).sum(axis=0)
        count += flat.shape[0]
    if count == 0:
        raise ValueError("no images supplied for normalisation statistics")
    mean = sums / count
    var = np.maximum(sums_sq / count - mean**2, 1e-12)
    return mean.tolist(), np.sqrt(var).tolist()
