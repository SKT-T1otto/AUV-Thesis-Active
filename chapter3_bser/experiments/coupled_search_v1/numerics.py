"""Finite-model helpers; no environment objects or global random generators."""
import hashlib
import json
import time
import numpy as np
from core.env.target_motion import EPS


def locked(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def probabilities(value):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(a < 0) or a.sum() <= 0:
        raise ValueError('invalid finite-model probabilities')
    return a/a.sum()


def swept_distances(a0, a1, t0, t1):
    """Batch equivalent of core.target_motion.swept_relative_min_distance."""
    r = np.asarray(a0,dtype=float)-np.asarray(t0,dtype=float)
    delta = np.asarray(a1,dtype=float)-np.asarray(a0,dtype=float)-(np.asarray(t1,dtype=float)-np.asarray(t0,dtype=float))
    denominator = np.sum(delta*delta, axis=-1)
    tau = np.zeros_like(denominator)
    np.divide(-np.sum(r*delta, axis=-1), denominator, out=tau, where=denominator > EPS)
    return np.linalg.norm(r+np.clip(tau, 0., 1.)[..., None]*delta, axis=-1)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds, *, queries=None):
        self.started = time.perf_counter()
        self.deadline = self.started+float(seconds)
        self.query_limit = queries
        self.queries = 0

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise BudgetExceeded('wall_clock_budget')

    def query(self):
        self.check()
        if self.query_limit is not None and self.queries >= self.query_limit:
            raise BudgetExceeded('path_query_budget')
        self.queries += 1

    @property
    def elapsed(self):
        return time.perf_counter()-self.started
