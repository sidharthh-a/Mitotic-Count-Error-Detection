"""Regenerate the project-copy manifest that Colab notebook cell 5 verifies before training.

Cell 5 of ``notebooks/colab_pipeline.ipynb`` runs against a copy of this repository mounted
from Google Drive. That copy has gone stale before: an older
``midogcx/models/error_predictor.py`` let the config block's ``n_splits`` reach
``HistGradientBoostingRegressor``, which failed only after the source baseline had trained.
The cell therefore compares every file the run depends on against a sha256 manifest, and
this script rebuilds that manifest whenever the repository changes.

Usage::

    python scripts/write_notebook_manifest.py           # rewrite the manifest in place
    python scripts/write_notebook_manifest.py --check    # exit 1 if it is out of date

Not hashed on purpose: ``docs/`` and the top-level ``*.md`` (prose the run never reads, so a
documentation edit must not invalidate the manifest), ``notebooks/*.ipynb`` (Colab autosaves
execution state into the mounted copy, so its bytes change during a session),
``data/metadata/figshare_cache/`` (raw download cache, read by no pipeline step), and
``.git``, ``__pycache__``, ``*.pyc``.
Keep ``ROOTS``/``SKIP_*`` in sync with the walk in cell 5 of the notebook, and run
``scripts/verify_readiness.py`` before a Colab run: it fails when this manifest is stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_pipeline.ipynb"

ROOTS = ("midogcx", "scripts", "configs", "tests", "frontend", "data/splits", "data/metadata")
TOP_LEVEL = ("requirements.txt",)
EXTRA_FILES = ("outputs/metadata/normalization.json",)
SKIP_DIRS = {"__pycache__", "figshare_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".ipynb"}

START = 'MANIFEST = """'
END = '"""'


def manifest_files() -> list[Path]:
    """Files whose bytes the Colab run depends on, in a stable order."""
    files: list[Path] = []
    for root in ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            parts = path.relative_to(ROOT).parts
            if any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
                continue
            if path.name.startswith(".") or path.suffix in SKIP_SUFFIXES:
                continue
            files.append(path)
    for name in TOP_LEVEL + EXTRA_FILES:
        path = ROOT / name
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_lines() -> list[str]:
    return [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in manifest_files()]


def manifest_from_notebook(source: list[str]) -> tuple[int, int, list[str]]:
    """Locate the ``MANIFEST`` block and return its bounds and its entries."""
    start = next((i for i, line in enumerate(source) if line.rstrip("\n") == START), None)
    if start is None:
        raise SystemExit(f"no {START!r} block in {NOTEBOOK}")
    end = next((i for i in range(start + 1, len(source)) if source[i].rstrip("\n") == END), None)
    if end is None:
        raise SystemExit(f"unterminated MANIFEST block in {NOTEBOOK}")
    return start, end, [line.rstrip("\n") for line in source[start + 1 : end]]


def notebook_cell(nb: dict) -> list[str]:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and any(
            line.rstrip("\n") == START for line in cell["source"]
        ):
            return cell["source"]
    raise SystemExit(f"no MANIFEST block in any code cell of {NOTEBOOK}")


def parse(lines: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        entries[rel] = digest
    return entries


def check() -> int:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    recorded = parse(manifest_from_notebook(notebook_cell(nb))[2])
    current = parse(manifest_lines())

    changed = sorted(p for p in recorded if p in current and recorded[p] != current[p])
    missing = sorted(p for p in recorded if p not in current)
    added = sorted(p for p in current if p not in recorded)

    for label, paths in (("changed", changed), ("gone from the repo", missing), ("new in the repo", added)):
        if paths:
            print(f"manifest {label}: {', '.join(paths[:10])}" + (" ..." if len(paths) > 10 else ""))

    if changed or missing or added:
        print("notebook manifest is out of date; run: python scripts/write_notebook_manifest.py")
        return 1
    print(f"notebook manifest is current: {len(current)} files")
    return 0


def rewrite() -> int:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = notebook_cell(nb)
    start, end, _ = manifest_from_notebook(source)
    lines = manifest_lines()
    source[start : end + 1] = [START + "\n", *[f"{line}\n" for line in lines], END + "\n"]

    body = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    NOTEBOOK.write_text(body, encoding="utf-8")
    print(f"wrote {len(lines)} entries into {NOTEBOOK.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the manifest is stale")
    args = parser.parse_args(argv)
    return check() if args.check else rewrite()


if __name__ == "__main__":
    sys.exit(main())
