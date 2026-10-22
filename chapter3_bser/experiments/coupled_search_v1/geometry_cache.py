"""Bounded exact-value caches, owned by one ray snapshot or public session.

Eviction only discards memoized calculations. No evidence, geometry, random
draw, posterior, or budget check is represented by these caches.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass
import sys
import numpy as np
from core.env.public_events import ScanObservation


def array_key(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return (array.shape, array.dtype.str, array.tobytes(order='C'))


def box_key(box):
    return tuple(array_key(value) for value in box)


def key_bytes(value):
    """Conservative per-entry accounting; shared key objects are charged again."""
    total = sys.getsizeof(value)
    if isinstance(value, tuple):
        total += sum(key_bytes(item) for item in value)
    return total


def readonly(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


class ByteLRU:
    def __init__(self, max_bytes, max_items):
        self.max_bytes, self.max_items = max(0, int(max_bytes)), max(0, int(max_items))
        self.items = OrderedDict()
        self.bytes = 0
        self.peak_bytes = 0

    def get(self, key):
        row = self.items.get(key)
        if row is None:
            return None
        self.items.move_to_end(key)
        return row[0]

    def discard(self, key):
        row = self.items.pop(key, None)
        if row is not None:
            self.bytes -= row[1]

    def put(self, key, value, payload_bytes):
        size = key_bytes(key) + int(payload_bytes) + sys.getsizeof((value, 0))
        self.discard(key)
        if not self.max_items or size > self.max_bytes:
            return
        while self.items and (self.bytes + size > self.max_bytes or len(self.items) >= self.max_items):
            _, (_, removed) = self.items.popitem(last=False)
            self.bytes -= removed
        self.items[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)


@dataclass(frozen=True)
class PreparedScan:
    key: tuple
    origins: np.ndarray
    directions: np.ndarray
    limits: np.ndarray
    parallel: np.ndarray
    distances: np.ndarray
    hits: np.ndarray
    rows: np.ndarray

    @property
    def nbytes(self):
        return sum(array.nbytes + sys.getsizeof(array) for array in
                   (self.origins, self.directions, self.limits, self.parallel,
                    self.distances, self.hits, self.rows)) + sys.getsizeof(self)


def prepare_scan(scan, bounds, sensor_range, *, key=()):
    # Keep the original repeat/tile, divisions, boundary limits and row order.
    origins = np.repeat(np.asarray(scan.origins, dtype=float), len(scan.directions), axis=0)
    directions = np.tile(np.asarray(scan.directions, dtype=float), (len(scan.origins), 1))
    limits = np.full(len(origins), float(sensor_range))
    bounds = np.asarray(bounds, dtype=float)
    for axis in range(3):
        d = directions[:, axis]
        travel = np.full(len(d), np.inf)
        np.divide(np.where(d > 0, bounds[axis], 0.) - origins[:, axis], d,
                  out=travel, where=abs(d) > 1e-12)
        limits = np.minimum(limits, np.maximum(0., travel))
    distances = np.asarray(scan.distances).reshape(-1)
    hits = np.asarray(scan.hit_mask).reshape(-1)
    rows = np.column_stack((origins, directions, distances, hits))
    return PreparedScan(key, *(readonly(value) for value in
        (origins, directions, limits, abs(directions) <= 1e-12, distances, hits, rows)))


class ObservationCache:
    def __init__(self, history_capacity, *, prepared_bytes=16*1024**2,
                 compatibility_bytes=4*1024**2, compatibility_items=4096):
        self.prepared = ByteLRU(prepared_bytes, history_capacity)
        self.compatibility = ByteLRU(compatibility_bytes, compatibility_items)
        self.counts = Counter()

    def prepare(self, scan, bounds, sensor_range):
        key = (array_key(scan.origins, float), array_key(scan.directions, float),
               array_key(scan.distances), array_key(scan.hit_mask), tuple(scan.agent_ids),
               array_key(bounds, float), array_key(float(sensor_range)))
        value = self.prepared.get(key)
        if value is not None:
            self.counts['prepared_hits'] += 1
            return value
        self.counts['prepared_misses'] += 1
        value = prepare_scan(scan, bounds, sensor_range, key=key)
        self.prepared.put(key, value, value.nbytes)
        return value

    def compatible(self, geometry, scan, bounds, sensor_range, tolerance, evaluate):
        # Non-public sentinels/custom observations retain the evaluator's own
        # behavior (also used by analytic injection checks), without memoization.
        if type(scan) is not ScanObservation:
            return bool(evaluate(geometry, scan, bounds, sensor_range, tolerance))
        prepared = self.prepare(scan, bounds, sensor_range)
        key = (tuple(box_key(box) for box in geometry.boxes), prepared.key,
               array_key(float(tolerance)))
        value = self.compatibility.get(key)
        if value is not None:
            self.counts['compatibility_hits'] += 1
            return value
        self.counts['compatibility_misses'] += 1
        # Exceptions/timeouts never reach insertion; both completed booleans do.
        value = bool(evaluate(geometry, scan, bounds, sensor_range, tolerance, prepared=prepared))
        self.compatibility.put(key, value, sys.getsizeof(value))
        return value

    def summary(self):
        return dict(counts=dict(self.counts), prepared_entries=len(self.prepared.items),
                    compatibility_entries=len(self.compatibility.items),
                    prepared_bytes=self.prepared.bytes, prepared_peak_bytes=self.prepared.peak_bytes,
                    compatibility_bytes=self.compatibility.bytes,
                    compatibility_peak_bytes=self.compatibility.peak_bytes)
