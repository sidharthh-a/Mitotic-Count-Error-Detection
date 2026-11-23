"""Specimen-level cohort construction, validation and freezing.

Role separation is a hard requirement of the protocol: a specimen used to fit the source
detector can never be used to calibrate its threshold, to supply acquisition ranking, or
to assess adaptation. :func:`validate_cohort` enforces that disjointness instead of
trusting the caller.

The default cohort is the one proposed in the research blueprint (55 specimens, seed
20261026). It is validated against the real manifest rather than assumed correct, and
any metadata problem raises instead of being silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..utils import load_json, save_json, sha256_file, sha256_text
from .metadata import MidogIndex

#: Roles that participate in an experiment. ``target_pool`` is the only unlabelled (as
#: far as a selector is concerned) role: its annotations exist but are hidden from every
#: acquisition rule until a region is selected.
VALID_ROLES = (
    "source_train",
    "source_calib",
    "source_assess",
    "target_pool",
    "target_assess",
)

#: Which tumour type each role must contain, and how many specimens it must hold.
DEFAULT_ROLE_SPEC: dict[str, dict] = {
    "source_train": {"tumor": "human breast cancer", "n": 20},
    "source_calib": {"tumor": "human breast cancer", "n": 5},
    "source_assess": {"tumor": "human breast cancer", "n": 8},
    "target_pool": {"tumor": "human melanoma", "n": 12},
    "target_assess": {"tumor": "human melanoma", "n": 10},
}

#: The blueprint's proposed cohort, transcribed verbatim before validation.
BLUEPRINT_COHORT: dict[str, list[int]] = {
    "source_train": [3, 8, 23, 38, 40, 48, 60, 69, 75, 76, 79, 92, 98, 105, 112, 122, 126, 127, 132, 138],
    "source_calib": [16, 63, 99, 101, 103],
    "source_assess": [11, 25, 43, 67, 91, 114, 119, 135],
    "target_pool": [505, 506, 509, 511, 513, 525, 529, 530, 543, 550, 551, 553],
    "target_assess": [515, 518, 521, 523, 524, 527, 532, 536, 546, 549],
}

#: Specimens used by the historical Mac pilot. Already seen during development, so they
#: can never serve as pristine assessment data in this study.
HISTORICAL_PILOT_SLIDES = {1, 24, 34, 39, 72, 78, 94, 134, 139, 400, 401, 402, 404, 526, 535, 542, 548}


@dataclass
class Cohort:
    name: str
    seed: int
    roles: dict[str, list[int]]
    manifest_sha256: str = ""
    notes: str = ""
    role_metadata: dict[str, dict] = field(default_factory=dict)

    def role_of(self, slide_id: int) -> str | None:
        for role, ids in self.roles.items():
            if int(slide_id) in set(ids):
                return role
        return None

    @property
    def all_slide_ids(self) -> list[int]:
        return [sid for ids in self.roles.values() for sid in ids]

    def slides(self, role: str) -> list[int]:
        return list(self.roles.get(role, []))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "roles": {k: [int(s) for s in v] for k, v in self.roles.items()},
            "manifest_sha256": self.manifest_sha256,
            "notes": self.notes,
            "role_metadata": self.role_metadata,
        }


def validate_cohort(
    index: MidogIndex,
    roles: dict[str, list[int]],
    role_spec: dict[str, dict] | None = None,
    allow_historical_pilot: bool = False,
) -> dict:
    """Check a cohort against the manifest. Raises ``ValueError`` on any violation.

    Returns a per-role composition summary for the audit report.
    """
    spec = DEFAULT_ROLE_SPEC if role_spec is None else role_spec
    problems: list[str] = []

    unknown_roles = set(roles) - set(VALID_ROLES)
    if unknown_roles:
        problems.append(f"unknown roles: {sorted(unknown_roles)}")

    seen: dict[int, str] = {}
    for role, ids in roles.items():
        for sid in ids:
            if sid in seen:
                problems.append(f"slide {sid} appears in both {seen[sid]} and {role}")
            seen[sid] = role

    for role, ids in roles.items():
        if len(ids) != len(set(ids)):
            problems.append(f"role {role} contains duplicate slide ids")
        for sid in ids:
            if not index.eligible(int(sid)):
                problems.append(f"{role}: slide {sid} is not in the official manifest")

    if not allow_historical_pilot:
        overlap = sorted(set(seen) & HISTORICAL_PILOT_SLIDES)
        if overlap:
            problems.append(f"cohort reuses historical pilot specimens: {overlap}")

    summary: dict[str, dict] = {}
    for role, ids in roles.items():
        rows = index.slides[index.slides["slide_id"].isin(ids)]
        expected = spec.get(role, {})
        if expected.get("n") is not None and len(rows) != expected["n"]:
            problems.append(f"{role}: expected {expected['n']} specimens, found {len(rows)}")
        if expected.get("tumor"):
            bad = rows[rows["tumor"] != expected["tumor"]]["slide_id"].tolist()
            if bad:
                problems.append(f"{role}: slides {bad} are not '{expected['tumor']}'")
        not_train = rows[rows["official_split"] != "train"]["slide_id"].tolist()
        if not_train:
            problems.append(f"{role}: slides {not_train} are not in the official train split")

        summary[role] = {
            "n_slides": int(len(rows)),
            "n_positives": int(rows["n_positive"].sum()),
            "tumors": rows["tumor"].value_counts().to_dict(),
            "scanners": rows["scanner"].value_counts().to_dict(),
            "origins": rows["origin"].value_counts().to_dict(),
            "official_splits": rows["official_split"].value_counts().to_dict(),
        }

    if problems:
        raise ValueError("cohort validation failed:\n  - " + "\n  - ".join(problems))
    return summary


def build_cohort(
    index: MidogIndex,
    manifest_path: str | Path,
    roles: dict[str, list[int]] | None = None,
    *,
    name: str = "cohort_v1",
    seed: int = 20261026,
    notes: str = "",
    role_spec: dict[str, dict] | None = None,
    allow_historical_pilot: bool = False,
) -> Cohort:
    """Validate ``roles`` (default: the blueprint cohort) and wrap it as a Cohort."""
    roles = BLUEPRINT_COHORT if roles is None else roles
    summary = validate_cohort(
        index, roles, role_spec=role_spec, allow_historical_pilot=allow_historical_pilot
    )
    return Cohort(
        name=name,
        seed=int(seed),
        roles={k: [int(s) for s in v] for k, v in roles.items()},
        manifest_sha256=sha256_file(manifest_path),
        notes=notes,
        role_metadata=summary,
    )


def save_cohort(cohort: Cohort, path: str | Path) -> str:
    """Freeze the cohort to JSON and return the payload hash."""
    payload = cohort.to_dict()
    payload["payload_sha256"] = sha256_text(str(sorted(payload["roles"].items())))
    save_json(payload, path)
    return payload["payload_sha256"]


def load_cohort(path: str | Path) -> Cohort:
    """Load a frozen cohort, verifying its payload hash when one is recorded.

    The whole experiment is defined against one immutable split, so a cohort file that
    no longer matches its own hash must fail loudly instead of silently repartitioning
    specimens (and leaking assessment data into training without any error).
    """
    payload = load_json(path)
    expected = payload.get("payload_sha256")
    if expected:
        roles = {k: [int(s) for s in v] for k, v in payload["roles"].items()}
        actual = sha256_text(str(sorted(roles.items())))
        if actual != expected:
            raise ValueError(
                f"cohort {path} failed its integrity check: recorded {expected[:12]}..., "
                f"recomputed {actual[:12]}...; refusing to load a modified split"
            )
    return Cohort(
        name=payload.get("name", "unnamed"),
        seed=int(payload.get("seed", 0)),
        roles={k: [int(s) for s in v] for k, v in payload["roles"].items()},
        manifest_sha256=payload.get("manifest_sha256", ""),
        notes=payload.get("notes", ""),
        role_metadata=payload.get("role_metadata", {}),
    )


def role_frame(index: MidogIndex, cohort: Cohort) -> pd.DataFrame:
    """Per-specimen table with role, metadata and annotation counts (for the audit)."""
    mapping = {sid: role for role, ids in cohort.roles.items() for sid in ids}
    rows = index.slides[index.slides["slide_id"].isin(mapping)].copy()
    rows["role"] = rows["slide_id"].map(mapping)
    return rows.sort_values(["role", "slide_id"]).reset_index(drop=True)


def check_domain_confounding(index: MidogIndex, cohort: Cohort) -> pd.DataFrame:
    """Report scanner/origin composition per role.

    The blueprint requires that tumour-type effects not be presented as isolated: if the
    target domain is acquired on a single scanner, scanner and tumour are confounded.
    """
    frame = role_frame(index, cohort)
    return (
        frame.groupby(["role", "tumor", "scanner", "origin"]).size().rename("n_slides").reset_index()
    )
