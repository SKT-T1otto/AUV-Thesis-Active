"""Sharp pairwise bounds for a fixed public decision and common continuation policy.

Both mass vectors include a no-handoff bin with value interval [0, 0].
An optional total-variation radius is supplied per route. The returned bounds
are sharp for known masses under independent bin constraints, and conservative
when the mass radii are positive. Uses only the Python standard library.
"""
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from typing import Sequence


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


def success_difference(
    mass_a: Sequence[float],
    mass_b: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    within_bin_gap: Sequence[float],
    *,
    tv_a: float = 0.0,
    tv_b: float = 0.0,
) -> Interval:
    """Bound P(success | propose a) minus P(success | propose b); bin 0 is no handoff."""
    columns = (mass_a, mass_b, lower, upper, within_bin_gap)
    if not len(mass_a) or any(len(col) != len(mass_a) for col in columns):
        raise ValueError("Expected nonempty columns with matching bin order and length")
    for col in (*columns, (tv_a, tv_b)):
        if any(not isfinite(x) or not 0 <= x <= 1 for x in col):
            raise ValueError("Probabilities, gap bounds and TV radii must be in [0, 1]")
    for mass in (mass_a, mass_b):
        if not isclose(fsum(mass), 1, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Each mass vector, including no handoff, must sum to one")
    if (lower[0], upper[0]) != (0, 0):
        raise ValueError("No-handoff bin must have zero success value")
    low_terms = []
    high_terms = []
    for a, b, lo, hi, gap in zip(*columns):
        if lo > hi:
            raise ValueError("Inverted value interval")
        difference = a - b
        allowance = min(a, b) * min(hi - lo, gap)
        low_terms.append(min(difference * lo, difference * hi) - allowance)
        high_terms.append(max(difference * lo, difference * hi) + allowance)
    return Interval(max(-1.0, fsum(low_terms) - tv_a - tv_b),
                    min(1.0, fsum(high_terms) + tv_a + tv_b))


def check_against_vertices():
    """Independent polygon-vertex check for empty, partial and full sharing."""
    from itertools import product

    checked = 0
    for a, b, lo, hi, gap in product(
            (0, .2, .5, 1), (0, .2, .5, 1), (0, .2), (.5, 1), (0, .125, .5, 1)):
        width = min(gap, hi - lo)
        vertices = [(lo, lo), (hi, hi), (lo, lo + width), (lo + width, lo),
                    (hi, hi - width), (hi - width, hi)]
        attainable = [a * x - b * y for x, y in vertices]
        result = success_difference((1-a, a), (1-b, b), (0, lo), (0, hi), (0, gap))
        assert isclose(result.lower, min(attainable), abs_tol=1e-12)
        assert isclose(result.upper, max(attainable), abs_tol=1e-12)
        checked += 1
    return checked


if __name__ == "__main__":
    count = check_against_vertices()
    a, b = (.10, .45, .15, .30), (.10, .25, .35, .30)
    lo, hi = (0, .65, .10, 0), (0, .80, .30, 1)
    for gap, expected in [(0.0, (.07, .14)), (.10, (.04, .17)), (.25, (-.005, .215))]:
        actual = success_difference(a, b, lo, hi, (0, 0, 0, gap))
        assert isclose(actual.lower, expected[0], abs_tol=1e-12)
        assert isclose(actual.upper, expected[1], abs_tol=1e-12)
        print(f"gap={gap:.2f}  lower={actual.lower:.4f}  upper={actual.upper:.4f}")
    print(f"{count} vertex comparisons and 3 four-bin examples passed")
