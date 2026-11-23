"""Download only the specimens required by a frozen cohort.

Nothing is downloaded without an explicit invocation, and the script refuses to start if
the destination cannot hold the files. The full MIDOG++ collection is 65 GiB; the frozen
55-specimen cohort is about 7.5 GiB.

    python scripts/download_specimens.py --dry-run
    python scripts/download_specimens.py --only target_pool
    python scripts/download_specimens.py                    # whole cohort

Destinations are resolved like every other script: ``$MIDOG_DATA_ROOT`` first, then
``project.data_root`` in config, then ``data/images``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from midogcx.config import load_config, resolve_data_root, resolve_path, resolve_specimen_path
from midogcx.data.metadata import slide_to_filename
from midogcx.data.splits import load_cohort
from midogcx.utils import get_logger

INVENTORY = "dataset_inventory.csv"
CHUNK = 1 << 20
MIN_FREE_MARGIN_GB = 2.0


def inventory_by_slide(meta_dir: Path) -> dict[str, dict]:
    path = meta_dir / INVENTORY
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python scripts/audit_dataset_inventory.py` first"
        )
    frame = pd.read_csv(path)
    out = {}
    for _, row in frame.iterrows():
        name = str(row.get("file_name"))
        if name.endswith(".tiff") and isinstance(row.get("download_url"), str):
            out[name] = {"url": row["download_url"], "size": int(row["size_bytes"])}
    return out


def download(url: str, dest: Path, expected_size: int, log) -> None:
    """Stream to a ``.part`` file then rename, so an interrupted run never leaves a
    truncated TIFF that later looks complete."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
        if have and resp.status_code != 206:
            have = 0  # server ignored the range request; restart cleanly
            resp.close()
            tmp.unlink(missing_ok=True)
            with requests.get(url, stream=True, timeout=120) as fresh:
                _write_stream(fresh, tmp, 0, expected_size, log)
        else:
            resp.raise_for_status()
            _write_stream(resp, tmp, have, expected_size, log)

    got = tmp.stat().st_size
    if expected_size and got != expected_size:
        raise RuntimeError(f"size mismatch for {dest.name}: got {got}, expected {expected_size}")
    tmp.replace(dest)


def _write_stream(resp, tmp: Path, offset: int, expected: int, log) -> None:
    mode = "ab" if offset else "wb"
    written = offset
    last = time.time()
    with tmp.open(mode) as fh:
        for block in resp.iter_content(chunk_size=CHUNK):
            if not block:
                continue
            fh.write(block)
            written += len(block)
            if time.time() - last > 5:
                pct = 100 * written / expected if expected else 0
                log.info("    %5.1f%% (%d/%d MiB)", pct, written // CHUNK, expected // CHUNK)
                last = time.time()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these cohort roles")
    ap.add_argument("--dry-run", action="store_true", help="report sizes and exit")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    log = get_logger("download")
    cfg = load_config(args.config, args.set)
    data_root = resolve_data_root(cfg)
    meta_dir = resolve_path(cfg, "paths.manifest").parent
    cohort = load_cohort(resolve_path(cfg, "paths.cohort"))

    roles = args.only or list(cohort.roles)
    wanted = [sid for role in roles for sid in cohort.roles.get(role, [])]
    if not wanted:
        log.error("no specimens selected; check --only %s", roles)
        return 2

    inventory = inventory_by_slide(meta_dir)
    plan = []
    for sid in wanted:
        name = slide_to_filename(sid)
        entry = inventory.get(name)
        if entry is None:
            log.warning("no download URL for %s; skipping", name)
            continue
        plan.append((sid, name, entry))

    total = sum(e["size"] for _, _, e in plan)
    log.info("plan: %d specimens, %.2f GiB -> %s", len(plan), total / 1024**3, data_root)

    # Query the nearest existing ancestor so a dry run never creates directories.
    probe = data_root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    log.info("free space at destination (%s): %.2f GiB", probe, free / 1024**3)
    fits = total <= free - MIN_FREE_MARGIN_GB * 1024**3

    if args.dry_run:
        for _sid, name, entry in plan:
            print(f"  {name}  {entry['size'] / 1024**3:6.2f} GiB")
        log.info(
            "dry run only; nothing downloaded. Destination %s %s the required %.2f GiB + %.1f GiB margin.",
            data_root,
            "can hold" if fits else "CANNOT hold",
            total / 1024**3,
            MIN_FREE_MARGIN_GB,
        )
        return 0

    if not fits:
        log.error(
            "refusing to download: need %.2f GiB plus %.1f GiB margin, only %.2f GiB free. "
            "Set MIDOG_DATA_ROOT to a larger disk.",
            total / 1024**3,
            MIN_FREE_MARGIN_GB,
            free / 1024**3,
        )
        return 3

    data_root.mkdir(parents=True, exist_ok=True)

    for i, (sid, name, entry) in enumerate(plan, 1):
        dest = resolve_specimen_path(data_root, name)
        if dest.exists() and dest.stat().st_size == entry["size"]:
            log.info("[%d/%d] %s already present", i, len(plan), name)
            continue
        log.info("[%d/%d] downloading %s (%.2f GiB)", i, len(plan), name, entry["size"] / 1024**3)
        download(entry["url"], dest, entry["size"], log)

    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
