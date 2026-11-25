"""Conditional first-registration model for one simultaneous departure.

All time windows and continuation-equivalence certificates are assumptions
made before an assignment. An ordinary candidate endpoint supplies neither.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Window:
    """Potential registration step if chosen and not preempted.

    A finite latest is a guaranteed opportunity by that step, conditional
    on the stated scenario. None permits no registration at all.
    """
    earliest: int
    latest: int | None = None

    def __post_init__(self):
        if type(self.earliest) is not int or self.earliest < 1:
            raise ValueError("registration starts at physical step one")
        if self.latest is not None and (type(self.latest) is not int or self.latest < self.earliest):
            raise ValueError("inverted or invalid registration window")


def event_rows(windows, certificates, *, horizon, hold):
    """Map each joint option to a conditional success variable; None means zero.

    certificates[arm] identifies an equal law of receipt/termination and
    continuation success, conditional on this scenario and winning option.
    It must cover the executor, delivered packet, relevant team state and
    remaining time. A shared finder or overlapping time windows do not suffice.
    """
    if type(hold) is not int or type(horizon) is not int or not 1 <= hold <= horizon:
        raise ValueError("invalid native hold/horizon")
    if not windows or any(not menu for menu in windows):
        raise ValueError("expected at least one option on every active boat")
    arms = list(product(*(range(len(menu)) for menu in windows)))
    if set(certificates) - set(arms) or any(not isinstance(v, str) or not v for v in certificates.values()):
        raise ValueError("invalid continuation-equivalence certificate")
    last_discovery = horizon - hold  # receipt at discovery+1; hold starts at receipt
    result = {}
    for arm in arms:
        selected = [menu[choice] for menu, choice in zip(windows, arm)]
        latest = min((w.latest for w in selected if w.latest is not None), default=horizon + 1)
        possible = tuple(i for i, w in enumerate(selected) if w.earliest <= min(latest, last_discovery))
        if not possible:
            value = None
        elif len(possible) == 1 and latest <= last_discovery and arm in certificates:
            winner, = possible
            value = ("receipt", winner, arm[winner], certificates[arm])
        else:
            value = ("private", arm)
        result[arm] = value
    return result


def contrast_bounds(scenarios, contrast, weights=None):
    """Sharp contrast interval for the induced model of free values in [0,1].

    Scenarios have an arm-independent mixture. Unknown mixture weights allow
    any common distribution. Values may vary independently across scenarios.
    """
    if not scenarios or not contrast or not scenarios[0]:
        raise ValueError("empty model or contrast")
    arms = set(scenarios[0])
    if not set(contrast) <= arms or any(set(rows) != arms for rows in scenarios):
        raise ValueError("inconsistent assignment domain")
    ranges = []
    for rows in scenarios:
        coefficients = defaultdict(Fraction)
        for arm, c in contrast.items():
            if rows[arm] is not None:
                coefficients[rows[arm]] += Fraction(c)
        ranges.append((sum(min(0, c) for c in coefficients.values()),
                       sum(max(0, c) for c in coefficients.values())))
    if weights is None:
        return min(lo for lo, _ in ranges), max(hi for _, hi in ranges)
    weights = tuple(map(Fraction, weights))
    if len(weights) != len(ranges) or min(weights) < 0 or sum(weights) != 1:
        raise ValueError("expected probabilities summing to one")
    return tuple(sum(w * r[i] for w, r in zip(weights, ranges)) for i in (0, 1))


def fourth_arm_interval(scenarios, measured, weights=None):
    """Bound Q11 using three simultaneous mean intervals; boat 2 stays on zero."""
    a, b, c, d = ((i,j,0) for i,j in product(range(2), repeat=2))
    if set(measured) != {a,b,c} or any(not 0 <= lo <= hi <= 1 for lo, hi in measured.values()):
        raise ValueError("expected three simultaneous probability intervals")
    lower, upper = contrast_bounds(scenarios, {a:1, b:-1, c:-1, d:1}, weights)
    lo = max(0, lower + measured[b][0] + measured[c][0] - measured[a][1])
    hi = min(1, upper + measured[b][1] + measured[c][1] - measured[a][0])
    if lo > hi:
        raise ValueError("measured intervals conflict with the event relation")
    return lo, hi


def standby_contrasts(menus, standby, radii):
    """Development hypothesis: one synchronous endpoint scan on a public lattice.

    Uniform static target over M20 cell centers, stationary executor, common
    nonincreasing completion probability in target-to-standby distance. These
    assumptions describe the proxy only, not the native moving-target mission.
    """
    if len(menus) != 3 or any(len(points) < needed for points, needed in zip(menus, (2, 2, 1))):
        raise ValueError("two options on boats 0/1 and option zero on boat 2 required")
    standby, radii = np.asarray(standby, dtype=float), np.asarray(radii, dtype=float)
    if standby.shape != (3,) or radii.shape != (3,) or not np.isfinite(standby).all() or not (
        np.isfinite(radii).all() and np.all(radii > 0)
    ):
        raise ValueError("invalid public geometry")
    sites = np.indices((20, 20, 8)).reshape(3, -1).T + 0.5
    distance_sq = np.sum((sites - standby) ** 2, axis=1)
    order = np.argsort(distance_sq)
    # Equal distances must share the same completion value.
    ends = np.r_[np.diff(distance_sq[order]) != 0, True]
    hits = []
    for points, n, r in zip(menus, (2, 2, 1), radii):
        centers = np.asarray(points[:n], dtype=float)
        if centers.shape != (n, 3) or not np.isfinite(centers).all():
            raise ValueError("invalid local offers")
        hits.append([np.sum((sites - p) ** 2, axis=1) <= r*r for p in centers])
    result = {}
    for b in (0, 1):
        peer = hits[1][b] | hits[2][0]
        delta = (peer | hits[0][1]).astype(int) - (peer | hits[0][0]).astype(int)
        prefix = np.cumsum(delta[order])[ends]
        result[str(b)] = dict(n_sites=len(sites), min_prefix=min(0, int(prefix.min())),
                              max_prefix=max(0, int(prefix.max())), constant_value=int(prefix[-1]))
    return result
