"""Bounds on the difference in mission success probabilities for two routes."""

from dataclasses import dataclass
from math import fsum, isclose
from numbers import Real
from typing import Sequence


NO_HANDOFF = "no_handoff"


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


def _unit_value(value: float, field: str) -> float:
    # This comparison also rejects NaN and both infinities.
    if not isinstance(value, Real) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1], got {value!r}")
    return float(value)


def _column(values: Sequence[float], field: str, bin_ids: Sequence[str]) -> tuple[float, ...]:
    if len(values) != len(bin_ids):
        raise ValueError(f"{field} has {len(values)} entries for {len(bin_ids)} bins")
    return tuple(_unit_value(value, f"{field}[{name}]") for name, value in zip(bin_ids, values))


def success_difference(
    mass_a: Sequence[float],
    mass_b: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    within_bin_gap: Sequence[float],
    *,
    bin_ids: Sequence[str],
    tv_a: float = 0.0,
    tv_b: float = 0.0,
) -> Interval:
    """Bound P(success | route A) - P(success | route B) at the same decision.

    All five vectors follow bin_ids. The required 'no_handoff' bin has
    conditional success interval [0, 0], including when its mass is zero.
    In each bin, both conditional success rates lie in [lower, upper] and
    can differ by at most within_bin_gap under the common continuation policy.

    For known masses the extrema are attainable with independent constraints
    across bins. For estimated masses, tv_a and tv_b bound the respective
    total-variation distances. Each can change the expectation of a [0, 1]
    success rate by at most its radius, giving a conservative interval.
    """
    if isinstance(bin_ids, str) or any(not isinstance(name, str) or not name for name in bin_ids):
        raise ValueError("bin_ids must be a sequence of nonempty string identifiers")
    if NO_HANDOFF not in bin_ids:
        raise ValueError(f"bin_ids must include the '{NO_HANDOFF}' bin")
    if len(set(bin_ids)) != len(bin_ids):
        raise ValueError("bin_ids must be unique, including 'no_handoff'")

    mass_a = _column(mass_a, "mass_a", bin_ids)
    mass_b = _column(mass_b, "mass_b", bin_ids)
    for field, mass in (("mass_a", mass_a), ("mass_b", mass_b)):
        total = fsum(mass)
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{field} must sum to 1 including '{NO_HANDOFF}', got {total!r}")
    lower = _column(lower, "lower", bin_ids)
    upper = _column(upper, "upper", bin_ids)
    within_bin_gap = _column(within_bin_gap, "within_bin_gap", bin_ids)
    radius = _unit_value(tv_a, "tv_a") + _unit_value(tv_b, "tv_b")

    lower_terms = []
    upper_terms = []
    for name, a, b, lo, hi, gap in zip(bin_ids, mass_a, mass_b, lower, upper, within_bin_gap):
        if lo > hi:
            raise ValueError(f"inverted interval for '{name}': lower {lo} exceeds upper {hi}")
        if name == NO_HANDOFF and (lo != 0.0 or hi != 0.0):
            raise ValueError(f"'{NO_HANDOFF}' must have success interval [0, 0]")
        # Only the mass common to both routes can contribute a within-bin
        # discrepancy. The excess mass takes one of the interval endpoints.
        excess = a - b
        allowance = min(a, b) * min(gap, hi - lo)
        lower_terms.append(min(excess * lo, excess * hi) - allowance)
        upper_terms.append(max(excess * lo, excess * hi) + allowance)

    # Clip both endpoints, including roundoff allowed by the unit-sum check.
    return Interval(
        min(1.0, max(-1.0, fsum(lower_terms) - radius)),
        min(1.0, max(-1.0, fsum(upper_terms) + radius)),
    )
