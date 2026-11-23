"""Freeze and validate the specimen-level cohort.

    python scripts/build_cohort.py
    python scripts/build_cohort.py --set paths.cohort=data/splits/cohort_v1.json

The script refuses to write a cohort that fails validation (wrong tumour type, specimen
outside the manifest, duplicate use, official-test leakage, or reuse of historical pilot
specimens). It also prints the scanner/origin composition so domain confounding is
visible at freeze time rather than discovered in the discussion section.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from midogcx.config import PROJECT_ROOT, load_config, resolve_path
from midogcx.data.metadata import build_index
from midogcx.data.splits import build_cohort, check_domain_confounding, load_cohort, save_cohort
from midogcx.utils import get_logger, load_yaml, save_json, sha256_text

DEFAULT_ROLES = PROJECT_ROOT / "configs" / "cohort_v1.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="base config yaml")
    ap.add_argument("--roles", default=str(DEFAULT_ROLES), help="cohort roles yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the frozen cohort already on disk matches the roles file; never writes",
    )
    args = ap.parse_args(argv)

    log = get_logger("build_cohort")
    cfg = load_config(args.config, args.set)
    manifest = resolve_path(cfg, "paths.manifest")
    annotations = resolve_path(cfg, "paths.annotations")
    cohort_path = resolve_path(cfg, "paths.cohort")

    log.info("loading metadata ...")
    index = build_index(manifest, annotations)

    roles_cfg = load_yaml(args.roles)
    roles = roles_cfg["roles"]
    seed = int(roles_cfg.get("seed", cfg["project"]["seed"]))
    name = roles_cfg.get("name", "cohort_v1")

    log.info("validating %d roles / %d specimens ...", len(roles), sum(len(v) for v in roles.values()))
    cohort = build_cohort(
        index,
        manifest,
        roles,
        name=name,
        seed=seed,
        notes=(
            "Blueprint cohort, validated against the official manifest. Official training "
            "records used for development; not the official test set."
        ),
    )
    if args.check:
        # A remote runner must never rewrite the frozen cohort: the whole experiment is
        # defined against one immutable split, and a silent regeneration would invalidate
        # every previously reported number.
        if not cohort_path.exists():
            log.error("no cohort at %s; run without --check first", cohort_path)
            return 2
        existing = load_cohort(cohort_path)
        rebuilt = cohort.to_dict()["roles"]
        current = {k: [int(s) for s in v] for k, v in existing.roles.items()}
        if current != rebuilt:
            differing = sorted(set(current) | set(rebuilt))
            log.error(
                "frozen cohort does not match %s; refusing to proceed. Roles: %s",
                args.roles,
                [r for r in differing if current.get(r) != rebuilt.get(r)],
            )
            return 3
        digest = sha256_text(str(sorted(rebuilt.items())))
        log.info("cohort verified unchanged (%s), nothing written", digest[:12])
        return 0

    digest = save_cohort(cohort, cohort_path)
    log.info("wrote %s (roles digest %s...)", cohort_path, digest[:12])

    confounding = check_domain_confounding(index, cohort)
    print("\nRole composition")
    print(confounding.to_string(index=False))

    print("\nPer-role summary")
    for role, meta in cohort.role_metadata.items():
        print(
            f"  {role:14s} n={meta['n_slides']:2d}  positives={meta['n_positives']:4d}  "
            f"scanners={meta['scanners']}"
        )

    save_json(
        {"cohort": cohort.to_dict(), "confounding": confounding.to_dict(orient="records")},
        cohort_path.with_name("cohort_v1_summary.json"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
