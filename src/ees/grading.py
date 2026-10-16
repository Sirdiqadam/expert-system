"""Grade bands and boundary helpers (shared by pipeline, validator, eval)."""

from __future__ import annotations

# (grade, minimum fraction of max_total)
GRADE_BANDS: list[tuple[str, float]] = [
    ("A", 0.80),
    ("B", 0.70),
    ("C", 0.60),
    ("D", 0.50),
    ("E", 0.40),
    ("F", 0.00),
]


def grade_for(total: float, max_total: float) -> str:
    """Return the letter grade for *total* out of *max_total*."""

    if max_total <= 0:
        return "F"
    fraction = total / max_total
    for grade, floor in GRADE_BANDS:
        if fraction >= floor:
            return grade
    return "F"


def boundary_marks(max_total: float) -> list[float]:
    """The mark values that separate one grade from the next."""

    return [round(floor * max_total, 4) for _grade, floor in GRADE_BANDS if floor > 0]
