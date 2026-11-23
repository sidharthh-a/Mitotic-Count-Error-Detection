"""Configuration loading and path resolution.

All scripts take ``--config`` (optionally layered with ``--set key=value`` overrides) so
that every experiment is reproducible from a file checked into ``configs/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import deep_update, get_by_path, load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def _parse_scalar(text: str) -> Any:
    """Parse a ``--set`` value into bool/int/float/None/list/str."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_parse_scalar(p) for p in inner.split(",")] if inner else []
    return text


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key.sub=value`` overrides, creating nested keys as needed."""
    out = cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_scalar(raw)
    return out


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    extra_files: list[str | Path] | None = None,
) -> dict:
    """Load the base config, optionally layer extra files, then apply overrides."""
    base = load_yaml(path or DEFAULT_CONFIG)
    for extra in extra_files or []:
        base = deep_update(base, load_yaml(extra))
    if overrides:
        base = apply_overrides(base, overrides)
    return base


def resolve_path(cfg: dict, key: str, default: str | None = None) -> Path:
    """Resolve a path from config; relative paths are relative to the project root."""
    raw = get_by_path(cfg, key, default)
    if raw is None:
        raise KeyError(f"config key {key!r} is not set")
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def resolve_data_root(cfg: dict) -> Path:
    """Resolve the directory that holds specimen TIFFs.

    Priority: ``$MIDOG_DATA_ROOT`` > ``project.data_root`` in config > ``data/images``.
    The environment variable exists because the project drive is too small for the
    cohort (see docs/ENVIRONMENT_AUDIT.md), so images may live on another disk.
    """
    env = os.environ.get("MIDOG_DATA_ROOT")
    if env:
        return Path(env)
    raw = get_by_path(cfg, "project.data_root")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return PROJECT_ROOT / "data" / "images"


def resolve_specimen_path(data_root: Path, file_name: str) -> Path:
    """Specimen TIFFs are stored flat as ``<data_root>/<NNN>.tiff``."""
    return Path(data_root) / file_name


def config_fingerprint(cfg: dict) -> str:
    """Stable hash of a config, used to prove two runs shared settings."""
    import json

    from .utils import sha256_text

    return sha256_text(json.dumps(cfg, sort_keys=True, default=str))
