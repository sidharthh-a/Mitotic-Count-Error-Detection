"""Build a complete file/size inventory of the official MIDOG++ Figshare collection.

The research blueprint requires an exact storage audit before any download:
"Query exact download sizes for that cohort and measure actual free space."

This script queries the public Figshare API (collection 6615571), caches every article
response under ``data/metadata/figshare_cache/`` so re-runs are offline, and writes:

    data/metadata/dataset_inventory.csv   one row per downloadable file
    data/metadata/dataset_inventory.json  joined manifest/annotation/metadata summary

Usage (from the project root)::

    python scripts/audit_dataset_inventory.py
    python scripts/audit_dataset_inventory.py --workers 8 --refresh
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

COLLECTION_ID = 6615571
API = "https://api.figshare.com/v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_DIR = PROJECT_ROOT / "data" / "metadata"
CACHE_DIR = META_DIR / "figshare_cache"

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def fetch_json(
    url: str,
    cache_path: Path,
    refresh: bool,
    timeout: int = 60,
    attempts: int = 6,
) -> dict | list:
    """Return cached JSON for ``url``, downloading it only when needed.

    Figshare returns 403/429 when the public API is hit too fast, so transient
    failures are retried with exponential backoff plus jitter.
    """
    if cache_path.exists() and not refresh:
        with cache_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            payload = resp.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            return payload
        except Exception as exc:  # noqa: BLE001 - deliberate retry wrapper
            last_exc = exc
            if attempt == attempts - 1:
                break
            time.sleep((2 ** attempt) * 0.5 + random.random() * 0.5)
    raise RuntimeError(f"failed to fetch {url}: {last_exc}")


def get_collection_articles(refresh: bool) -> list[dict]:
    url = f"{API}/collections/{COLLECTION_ID}/articles?page_size=1000"
    payload = fetch_json(url, CACHE_DIR / "_collection_articles.json", refresh)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected collection payload type: {type(payload)}")
    return payload


def get_article(article_id: int, refresh: bool) -> dict:
    return fetch_json(
        f"{API}/articles/{article_id}",
        CACHE_DIR / f"article_{article_id}.json",
        refresh,
    )


def build_file_rows(articles: list[dict], refresh: bool, workers: int) -> list[dict]:
    """Fetch every article detail and flatten its files into one row per file."""
    ids = [int(a["id"]) for a in articles]

    def work(article_id: int) -> list[dict]:
        try:
            art = get_article(article_id, refresh)
        except Exception as exc:  # network flake: record, do not abort the audit
            _log(f"  ! article {article_id} failed: {exc}")
            return [
                {
                    "article_id": article_id,
                    "article_title": None,
                    "file_name": None,
                    "size_bytes": None,
                    "download_url": None,
                    "error": str(exc),
                }
            ]
        rows = []
        for f in art.get("files", []):
            rows.append(
                {
                    "article_id": article_id,
                    "article_title": art.get("title"),
                    "file_name": f.get("name"),
                    "size_bytes": f.get("size"),
                    "download_url": f.get("download_url"),
                    "error": None,
                }
            )
        return rows

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(work, ids):
            rows.extend(part)
            done += 1
            if done % 50 == 0:
                _log(f"  fetched {done}/{len(ids)} articles")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4, help="parallel Figshare requests")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-download")
    args = ap.parse_args(argv)

    META_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Querying Figshare collection {COLLECTION_ID} ...")
    articles = get_collection_articles(args.refresh)
    print(f"  {len(articles)} articles")

    print("Fetching article details (cached) ...")
    rows = build_file_rows(articles, args.refresh, args.workers)

    ok = [r for r in rows if r["size_bytes"] is not None]
    print(f"  {len(ok)} downloadable files, {len(rows) - len(ok)} unresolved")

    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["file_name"], na_position="last")
    csv_path = META_DIR / "dataset_inventory.csv"
    df.to_csv(csv_path, index=False)

    total = int(df["size_bytes"].fillna(0).sum())
    sizes_mib = df["size_bytes"].dropna() / 1024**2
    summary = {
        "collection_id": COLLECTION_ID,
        "n_articles": len(articles),
        "n_files": int(len(ok)),
        "n_unresolved": int(len(rows) - len(ok)),
        "total_size_bytes": total,
        "total_size_gib": round(total / 1024**3, 2),
        "biggest_file_mib": round(float(sizes_mib.max()), 1) if len(sizes_mib) else None,
        "median_file_mib": round(float(sizes_mib.median()), 1) if len(sizes_mib) else None,
        "smallest_file_mib": round(float(sizes_mib.min()), 2) if len(sizes_mib) else None,
    }
    with (META_DIR / "dataset_inventory_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
