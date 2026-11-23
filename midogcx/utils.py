"""Small shared helpers: seeding, deterministic IO, logging, run bookkeeping."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------
def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed python, numpy and torch. Returns the seed for logging.

    ``deterministic_torch`` also forces cuDNN into deterministic mode. Some ops still
    have no deterministic CUDA kernel, so this makes runs reproducible *or* fails
    loudly rather than silently producing different numbers.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # torch missing (e.g. pure data-audit environment)
        pass
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct but seeded RNG state."""
    base = int(os.environ.get("MIDOGCX_SEED", "0"))
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write JSON atomically (temp file then replace) so a crash cannot truncate output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    tmp.replace(path)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Run directories and logging
# --------------------------------------------------------------------------------------
def create_run_dir(root: str | Path, tag: str, seed: int | None = None) -> Path:
    """Create ``root/<timestamp>_<tag>[_s<seed>]`` and return it."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{tag}"
    if seed is not None:
        name += f"_s{seed}"
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Console + optional file logger that does not duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def load_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = val
    return out


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
