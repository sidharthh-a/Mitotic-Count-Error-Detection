"""Annotation-area budget accounting.

The protocol compares strategies at an **equal unique annotated tissue area**, not an
equal patch count: two 512x512 regions can contain very different amounts of tissue, and
overlapping regions would double-count area. Candidate regions produced by
:mod:`midogcx.data.patches` are non-overlapping, so their tissue areas are additive; this
module still records every admitted region and rejects duplicates so the accounting is
explicit rather than assumed.

Policy when a region does not fit the remaining budget: it is **skipped**, and the next
ranked region is considered. Repeating until the budget is exhausted gives the same total
area for every strategy under a fixed budget, which is what the comparison requires.
Equal area is not equal expert time and must not be described as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AreaBudget:
    total_um2: float
    used_um2: float = 0.0
    admitted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_um2 <= 0:
            raise ValueError("budget must be positive")

    @property
    def remaining_um2(self) -> float:
        return max(0.0, self.total_um2 - self.used_um2)

    @property
    def utilisation(self) -> float:
        return self.used_um2 / self.total_um2 if self.total_um2 else 0.0

    def can_admit(self, region) -> bool:
        if region.region_id in self.admitted:
            return False
        return region.tissue_area_um2 <= self.remaining_um2 + 1e-9

    def admit(self, region) -> bool:
        if not self.can_admit(region):
            return False
        self.admitted.append(region.region_id)
        self.used_um2 += float(region.tissue_area_um2)
        return True

    def to_dict(self) -> dict:
        return {
            "total_mm2": self.total_um2 / 1e6,
            "used_mm2": self.used_um2 / 1e6,
            "utilisation": self.utilisation,
            "n_admitted": len(self.admitted),
            "n_skipped": len(self.skipped),
            "admitted": list(self.admitted),
            "skipped": list(self.skipped),
        }


def assert_disjoint_regions(regions) -> None:
    """Refuse to price a budget over regions that overlap on the same specimen.

    The region grid is non-overlapping by construction, so tissue area is additive. That
    is an assumption the budget arithmetic depends on, and an unverified assumption is
    how double counting enters a comparison: two overlapping regions would each be
    charged their full tissue area while annotating the same cells twice.

    Raises ``ValueError`` naming the offending pair, so a future change to the region
    generator cannot silently inflate every strategy's effective budget.
    """
    by_slide: dict[int, list] = {}
    for region in regions:
        by_slide.setdefault(int(region.slide_id), []).append(region)
    for sid, slide_regions in by_slide.items():
        ordered = sorted(slide_regions, key=lambda r: (r.y, r.x))
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if b.y >= a.y + a.height:
                    break  # sorted by y: no later region can overlap vertically
                if a.x < b.x + b.width and b.x < a.x + a.width:
                    raise ValueError(
                        f"regions overlap on specimen {sid}: {a.region_id} and {b.region_id}; "
                        "equal-area budgets require disjoint regions"
                    )


def budget_from_fraction(regions, fraction: float) -> AreaBudget:
    """Build a budget equal to ``fraction`` of the candidate pool's total tissue area."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be within (0, 1]")
    total = float(sum(r.tissue_area_um2 for r in regions))
    if total <= 0:
        raise ValueError("candidate pool has no tissue area")
    return AreaBudget(total_um2=total * fraction)


def select_under_budget(region_lookup: dict, ranking: list[tuple[str, float]], budget: AreaBudget):
    """Walk a ranking, admitting regions until the budget is exhausted.

    ``region_lookup`` maps ``region_id -> Region``. Regions that would overshoot are
    recorded in ``skipped`` and the walk continues, so a strategy cannot be penalised for
    the granularity of its ranking.
    """
    selected = []
    for rid, _score in ranking:
        region = region_lookup[rid]
        if budget.admit(region):
            selected.append(region)
        else:
            budget.skipped.append(rid)
    return selected
