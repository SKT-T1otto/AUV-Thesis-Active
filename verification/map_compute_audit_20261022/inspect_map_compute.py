"""Bounded offline audit; never trains, restores, or edits production sources.

Run from repository root with ``python -m verification.map_compute_audit_20261022.inspect_map_compute``.
Experimental kernels below are local comparison subjects, never runtime imports.
Each invocation writes a new results_<uuid>.json and refuses to overwrite it.
"""
import copy
import cProfile
import hashlib
import itertools
import json
from pathlib import Path
import platform
import pstats
import time
import uuid
from unittest.mock import patch

import numpy as np

from core.env.public_events import ScanObservation
from chapter3_bser.experiments.coupled_search_v1.core_config import CORE_DEFAULT
from chapter3_bser.experiments.coupled_search_v1.geometry_hypotheses import (
    GeometryHypothesis, GeometryHypothesisBank, RayConstraints, compatible,
)
from chapter3_bser.experiments.coupled_search_v1.numerics import Budget

ROOT = Path(__file__).resolve().parents[2]
BOUNDS = np.array([20., 20., 10.])
SENSOR_RANGE = 4.5
DIRECTIONS = np.array([v for v in itertools.product((-1., 0., 1.), repeat=3) if any(v)])
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1)[:, None]


def scalar_entry(origin, direction, box, limit):
    """Independent scalar slab oracle, including parallel rays and tangencies."""
    center, size = box
    lower, upper = np.asarray(center) - np.asarray(size)/2, np.asarray(center) + np.asarray(size)/2
    near, far = 0., np.inf
    for axis in range(3):
        if abs(direction[axis]) <= 1e-12:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return np.inf
            continue
        left = (lower[axis] - origin[axis])/direction[axis]
        right = (upper[axis] - origin[axis])/direction[axis]
        near = max(near, min(left, right))
        far = min(far, max(left, right))
    return near if near <= far and far >= 0 and near < limit else np.inf


def scan_at(origins, boxes):
    origins = np.asarray(origins, dtype=float)
    distances, hits = [], []
    for origin in origins:
        row, flags = [], []
        for direction in DIRECTIONS:
            limit = SENSOR_RANGE
            for axis in range(3):
                if abs(direction[axis]) > 1e-12:
                    travel = ((BOUNDS[axis] if direction[axis] > 0 else 0) - origin[axis])/direction[axis]
                    limit = min(limit, max(0., travel))
            best = min([limit] + [scalar_entry(origin, direction, box, limit) for box in boxes])
            row.append(float(best)); flags.append(bool(best < limit - 1e-10))
        distances.append(tuple(row)); hits.append(tuple(flags))
    return ScanObservation(tuple(map(tuple, origins)), tuple(map(tuple, DIRECTIONS)),
                           tuple(distances), tuple(hits), tuple(range(len(origins))))


def synthetic_history(count):
    # Measurements only: no environment, simulator steps, true research inputs,
    # model weights, or historical outputs are fed to the filter.
    boxes = (((10., 10., 5.), (1., 12., 8.)),)
    scans = []
    for step in range(count):
        origins = [(6.1 + .15*np.sin(step*.03+i),
                    4.1 + (step*.037 + i*2.6) % 11.5,
                    2.1 + i*1.6) for i in range(4)]
        scans.append(scan_at(origins, boxes))
    return scans


def axis_entry(rays, box):
    """Candidate only: same divisions and slab operations, fewer Nx3 temporaries.

    Deliberately no reciprocal multiplication, float32, JIT, or fastmath.
    Production RayConstraints.entry remains untouched.
    """
    center, size = map(np.asarray, box)
    lower, upper = center-size/2, center+size/2
    outside = np.zeros(len(rays.origins), dtype=bool)
    enter = np.full(len(rays.origins), -np.inf)
    leave = np.full(len(rays.origins), np.inf)
    for axis in range(3):
        origin, direction, parallel = rays.origins[:, axis], rays.directions[:, axis], rays.parallel[:, axis]
        outside |= parallel & ((origin < lower[axis]) | (origin > upper[axis]))
        first, second = np.zeros(len(origin)), np.zeros(len(origin))
        np.divide(lower[axis]-origin, direction, out=first, where=~parallel)
        np.divide(upper[axis]-origin, direction, out=second, where=~parallel)
        enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(first, second)))
        leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(first, second)))
    enter = np.maximum(0., enter)
    return np.where((~outside) & (enter <= leave) & (leave >= 0) & (enter < rays.limits), enter, np.inf)


def oracle_checks(scans):
    rays = RayConstraints(scans[:2], BOUNDS, SENSOR_RANGE, .25)
    rng = np.random.default_rng(72823)
    boxes = [(tuple(rng.uniform(0, BOUNDS)), tuple(rng.uniform(.01, 4., 3))) for _ in range(64)]
    boxes += [((6.1, 4.1, 2.1), (.1, .1, .1)), ((0., 0., 0.), (2., 2., 2.))]
    comparisons = 0
    for box in boxes:
        actual = rays.entry(box)
        expected = np.array([scalar_entry(o, d, box, limit) for o, d, limit in zip(rays.origins, rays.directions, rays.limits)])
        np.testing.assert_array_equal(actual, expected)
        candidate = axis_entry(rays, box)
        if actual.tobytes() != candidate.tobytes():
            raise AssertionError('candidate slab bytes differ')
        comparisons += len(actual)
    geometry = GeometryHypothesis((((10., 10., 5.), (1., 12., 8.)),))
    assert rays.matches(geometry)
    assert all(compatible(geometry, scan, BOUNDS, SENSOR_RANGE, .25) for scan in scans)
    duplicated = RayConstraints([scans[0], scans[0]], BOUNDS, SENSOR_RANGE, .25)
    single = RayConstraints([scans[0]], BOUNDS, SENSOR_RANGE, .25)
    assert duplicated.origins.tobytes() == single.origins.tobytes()
    assert duplicated.distances.tobytes() == single.distances.tobytes()
    return dict(scalar_ray_box_comparisons=comparisons, axis_candidate_byte_equal=True,
                duplicate_constraints_equal=True, all_256_synthetic_scans_match_witness=True)


def merge_counterexample():
    # Full public shape: four origins, 26 unit directions. Six thin walls
    # around each origin explain every short hit without enclosing a sensor.
    # This tests geometry constraints, not vehicle clearance feasibility.
    origins = [(1., 1.+i*2, 1.) for i in range(4)]
    boxes = []
    for origin in origins:
        for axis in range(3):
            for sign in (-1, 1):
                center = np.array(origin); center[axis] += sign*.02
                size = np.full(3, .1); size[axis] = .02
                boxes.append((tuple(center), tuple(size)))
    witness = GeometryHypothesis(tuple(boxes))
    scan = scan_at(origins, witness.boxes)
    valid = compatible(witness, scan, BOUNDS, SENSOR_RANGE, .25)
    assert valid and np.all(witness.free(scan.origins))
    bank = GeometryHypothesisBank(copy.deepcopy(CORE_DEFAULT),
        dict(bounds=BOUNDS, obstacle_sensor_range=SENSOR_RANGE), np.random.default_rng(17))
    try:
        bank.propose([scan], Budget(.75))
        outcome = 'accepted'
    except (ValueError, RuntimeError) as error:
        outcome = str(error)
    return dict(witness_valid=bool(valid), scan_shape=[4, 26], bank_outcome=outcome,
                scope='analytic close-wall fixture, not a physical episode or Linux failure reproduction')


def full_bank_candidate_checks(scans):
    """Compare generated worlds and the shared generator's subsequent state."""
    rows = []
    for count in (32, 64):
        outputs, duration, state = [], [], []
        for candidate in (False, True):
            bank = GeometryHypothesisBank(copy.deepcopy(CORE_DEFAULT),
                dict(bounds=BOUNDS, obstacle_sensor_range=SENSOR_RANGE), np.random.default_rng(17))
            start = time.perf_counter()
            if candidate:
                with patch.object(RayConstraints, 'entry', axis_entry):
                    maps = bank.propose(scans[:count], Budget(2.))
            else:
                maps = bank.propose(scans[:count], Budget(2.))
            duration.append(time.perf_counter()-start)
            outputs.append(tuple(m.boxes for m in maps))
            state.append(copy.deepcopy(bank.rng.bit_generator.state))
        assert outputs[0] == outputs[1]
        assert state[0] == state[1]
        rows.append(dict(scans=count, maps_equal=True, rng_state_equal=True,
            current_seconds=duration[0], axis_candidate_seconds=duration[1],
            speedup=duration[0]/duration[1], map_ids=[GeometryHypothesis(b).identity for b in outputs[0]]))
    return rows


def timings(scans):
    output = []
    rng = np.random.default_rng(711)
    boxes = [(tuple(rng.uniform(0, BOUNDS)), tuple(rng.uniform(.05, 3., 3))) for _ in range(24)]
    for count in (32, 128, 256):
        start = time.perf_counter()
        rays = RayConstraints(scans[:count], BOUNDS, SENSOR_RANGE, .25)
        build = time.perf_counter()-start
        timings_by_kernel = {}
        originals = [rays.entry(box) for box in boxes]
        assert all(a.tobytes() == axis_entry(rays, b).tobytes() for a, b in zip(originals, boxes))
        for name, kernel in [('current', lambda b: rays.entry(b)), ('axis_candidate', lambda b: axis_entry(rays, b))]:
            repeats = []
            for _ in range(3):
                start = time.perf_counter()
                for box in boxes:
                    kernel(box)
                repeats.append(time.perf_counter()-start)
            timings_by_kernel[name] = float(np.median(repeats))
        profile = cProfile.Profile()
        bank = GeometryHypothesisBank(copy.deepcopy(CORE_DEFAULT),
            dict(bounds=BOUNDS, obstacle_sensor_range=SENSOR_RANGE), np.random.default_rng(17))
        start = time.perf_counter()
        profile.enable()
        try:
            maps = bank.propose(scans[:count], Budget(.75))
            result = dict(outcome='completed', support_boxes=bank.support_boxes, map_ids=[m.identity for m in maps])
        except (ValueError, RuntimeError) as error:
            result = dict(outcome=str(error))
        finally:
            profile.disable()
        elapsed = time.perf_counter()-start
        entries = []
        for (filename, line, name), (primitive, calls, own, cumulative, _callers) in pstats.Stats(profile).stats.items():
            if 'coupled_search_v1' in filename:
                entries.append(dict(function=name, line=line, calls=calls, own_seconds=own, cumulative_seconds=cumulative))
        entries.sort(key=lambda x: x['cumulative_seconds'], reverse=True)
        output.append(dict(scans=count, distinct_rays=len(rays.origins), hits=int(rays.hits.sum()),
            constraint_build_seconds=build, kernel_24_boxes_median_seconds=timings_by_kernel,
            kernel_speedup=timings_by_kernel['current']/timings_by_kernel['axis_candidate'],
            candidate_byte_equal=True, profiled_propose_seconds=elapsed, profiled_propose=result,
            profile=entries[:10]))
    return output


def main():
    start = time.perf_counter()
    names = ['core/mapping/map_module.py', 'core/mapping/path_planner.py',
        'core/mapping/planning_graph.py', 'core/mapping/travel_cost_service.py',
        'chapter3_bser/experiments/coupled_search_v1/geometry_hypotheses.py',
        'chapter3_bser/experiments/coupled_search_v1/coupled_belief.py',
        'chapter3_bser/experiments/coupled_search_v1/target_particles.py',
        'chapter3_bser/experiments/coupled_search_v1/core_config.py']
    before = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    scans = synthetic_history(256)
    result = dict(schema='auv.offline_map_compute_audit.v1',
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        physical_episodes=0, physical_steps=0, model_updates=0, checkpoint_restores=0,
        source_sha256=before, oracle_checks=oracle_checks(scans),
        merge_counterexample=merge_counterexample(), timings=timings(scans),
        full_bank_candidate_checks=full_bank_candidate_checks(scans))
    after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    assert before == after
    result['production_sources_unchanged'] = True
    result['elapsed_seconds'] = time.perf_counter()-start
    path = Path(__file__).with_name('results_'+uuid.uuid4().hex[:12]+'.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(output=str(path), **{k: v for k, v in result.items() if k in
        ('oracle_checks', 'merge_counterexample', 'elapsed_seconds', 'full_bank_candidate_checks')}, timings=[{k: v for k, v in row.items()
        if k != 'profile'} for row in result['timings']]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
