"""Passive OLD runtime recorder. Never imported by policy/environment modules."""
from collections import deque
from contextlib import ExitStack, contextmanager
import dataclasses
import enum
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch

SCHEMA = 'auv.executor_trace.v1'
TIMING_KEYS = {'compile_seconds', 'max_compile_seconds'}
MAP_FIELDS = ('grid_size', 'cell_dx', 'cell_dy', 'cell_dz', 'z_range', 'eps',
    '_space_size_np', 'flat_xyz_centers', 'map_revision', 'grid_revision',
    'occupancy_probability', 'occupancy_logodds', 'known_free_mask',
    'known_occupied_mask', 'unknown_mask', 'valid_mask', 'belief_map',
    'last_map_changed_cell_count', 'last_planning_changed_cell_count',
    'planner_obstacle_clearance')
PHYSICS_FIELDS = ('_agent_pos', '_agent_vel', '_agent_acc', '_nav_targets',
    '_last_prior_acc', '_last_residual_acc', 'agent_finished')
CONTROL_FIELDS = ('dt', 'prior_kv_xy', 'prior_kv_z', 'prior_slow_radius_xy',
    'prior_slow_radius_z', 'prior_strength_search', 'prior_strength_executor',
    'residual_scale_search', 'residual_scale_executor', 'use_residual_prior', 'eps')
SPEC_FIELDS = ('name', 'a_xy_max', 'a_z_max', 'v_xy_max', 'v_z_max',
               'drag_xy', 'drag_z', 'buoyancy_bias', 'sensor_range')


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if torch.is_tensor(value):
        return plain(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, enum.Enum):
        return plain(value.value)
    if isinstance(value, torch.dtype):
        return {'torch_dtype': str(value)}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: plain(v) for k, v in value.items()}
        return {'typed_mapping': sorted([[plain(k), plain(v)] for k, v in value.items()], key=lambda x: repr(x[0]))}
    if isinstance(value, (list, tuple, deque)):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((plain(v) for v in value), key=repr)
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('unsupported diagnostic value: '+str(type(value)))


def encoded(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


class Blobs:
    def __init__(self, root, *, existing=False):
        self.root = Path(root)
        if not existing:
            self.root.mkdir(exist_ok=False)

    def put(self, value):
        data = encoded(value)
        name = hashlib.sha256(data).hexdigest()
        path = self.root/(name+'.json.gz')
        if path.exists():
            if gzip.decompress(path.read_bytes()) != data:
                raise ValueError('existing diagnostic blob changed')
        else:
            with path.open('xb') as stream:
                stream.write(gzip.compress(data, mtime=0))
        return name

    def get(self, name):
        if len(name) != 64 or any(c not in '0123456789abcdef' for c in name):
            raise ValueError('invalid diagnostic blob identity')
        data = gzip.decompress((self.root/(name+'.json.gz')).read_bytes())
        if hashlib.sha256(data).hexdigest() != name:
            raise ValueError('diagnostic blob changed')
        return json.loads(data)


def fields(owner, names):
    stored = vars(owner)
    return {name: plain(stored[name]) for name in names if name in stored}


def rng_hash():
    return digest((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def behavioral_summary(value):
    """Exclude only known wall timers; never exclude outcomes or event counts."""
    if isinstance(value, dict):
        return {k: behavioral_summary(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [behavioral_summary(v) for v in value]
    return value


def signature(runtime, transition=None):
    raw, provider = runtime.env.unwrapped, vars(runtime.provider)
    bridge = vars(runtime.bridge)
    return digest(dict(physical=fields(raw, PHYSICS_FIELDS), rng=rng_hash(),
        live_map=fields(raw.map_module, MAP_FIELDS), cached_state=runtime.state,
        provider_state=provider['cached'],
        refresh={k: v for k, v in provider.items() if k.startswith('_last') or k.endswith('_count')},
        guidance=runtime.guidance, allocation=runtime.controller.current_allocation,
        tracker=vars(runtime.bridge.path_tracker), observations=runtime.observations,
        planner_cache=vars(raw.map_module).get('_geodesic_cache', {}),
        safe_counts=runtime.safe_counts, effect_counts=runtime.effect_counts,
        guard={k: bridge[k] for k in ('counts', '_last', 'trace', '_executor_override') if k in bridge},
        transition=transition))


def snapshot(runtime, blobs):
    """Read existing arrays and read-only tracker; never construct/query a graph."""
    raw, provider = runtime.env.unwrapped, vars(runtime.provider)
    item = next(x for x in runtime.guidance.agent_assignments if x.agent_id == 3)
    guard = vars(runtime.bridge)
    return dict(step=int(runtime.state.step), guidance=plain(runtime.guidance),
        semantic_assignment=plain(runtime.controller.current_allocation.executor_assignment),
        executor_guidance=plain(item), tracker=plain(runtime.bridge.path_tracker.snapshot(3)),
        physical=fields(raw, PHYSICS_FIELDS),
        live_map_blob=blobs.put(fields(raw.map_module, MAP_FIELDS)),
        decision_state_blob=blobs.put(runtime.state),
        provider_cached_state_blob=blobs.put(provider['cached']),
        provider_refresh={k: plain(v) for k, v in provider.items() if k.startswith('_last') or k.endswith('_count')},
        allocator_audit_length=len(runtime.controller.allocator.audit),
        allocator_audit_tail=plain(runtime.controller.allocator.audit[-3:]),
        guard_last=plain(guard.get('_last')),
        path_tracking_threshold=float(runtime.bridge.path_tracker.threshold))


@contextmanager
def tap_method(owner, name, observer):
    """Forward original arguments once; restore the exact original lookup state."""
    original = getattr(owner, name)
    had_override = name in vars(owner)
    stored_override = vars(owner).get(name)

    def forwarded(*args, **kwargs):
        before = observer('before', args, kwargs, None)
        result = original(*args, **kwargs)
        observer('after', args, kwargs, (before, result))
        return result

    setattr(owner, name, forwarded)
    try:
        yield
    finally:
        if had_override:
            setattr(owner, name, stored_override)
        else:
            delattr(owner, name)


class Taps:
    """Observe actual pre-integration targets and existing controller results."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.physics = None
        self.controller = None
        self.guidance = None
        self.physics_calls = 0

    def physical(self, when, args, kwargs, result):
        raw = self.runtime.env.unwrapped
        if when == 'before':
            return dict(physical=fields(raw, PHYSICS_FIELDS), normalized_actions=plain(args[0]),
                        phase=str(vars(raw).get('_phase_before_step', 'unavailable')))
        before, _ = result
        self.physics_calls += 1
        self.physics = dict(before=before, after=fields(raw, PHYSICS_FIELDS),
                            collisions=plain(vars(raw).get('last_collision_records', [])))

    def controller_result(self, when, args, kwargs, result):
        if when == 'after':
            self.controller = plain(args[0])

    def compiled(self, when, args, kwargs, result):
        if when == 'after':
            _, context = result
            self.guidance = dict(allocation=plain(args[0]), context=plain(context))

    @contextmanager
    def installed(self):
        with ExitStack() as stack:
            stack.enter_context(tap_method(self.runtime.env.unwrapped, '_apply_agent_dynamics', self.physical))
            stack.enter_context(tap_method(self.runtime.env, 'observe_controller_result', self.controller_result))
            stack.enter_context(tap_method(self.runtime.bridge, 'compile_guidance', self.compiled))
            yield self


def run_episode(config, scene, output, *, instrumented):
    from chapter3_bser.learning.runtime import RepairedMissionRuntime
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.learning import map_contract
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    blobs = Blobs(output/'blobs') if instrumented else None
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    runtime, count, physical_steps = None, 0, 0
    signatures, elapsed = [], []
    try:
        runtime = RepairedMissionRuntime(config, scene, seed=int(scene['scenario_seed']), map_research=None)
        if runtime.map_research_session is not None:
            raise ValueError('passive executor recorder requires OLD map')
        raw = runtime.env.unwrapped
        write(output/'public_control.json', dict(control=fields(raw, CONTROL_FIELDS),
            bounds=plain(vars(raw)['space_size']),
            specs=[{k: plain(spec[k]) for k in SPEC_FIELDS} for spec in raw.agent_specs]))
        signatures.append(signature(runtime))
        taps = Taps(runtime)
        with ExitStack() as stack:
            if instrumented:
                stack.enter_context(taps.installed())
            stream = stack.enter_context((output/'trace.jsonl').open('x', encoding='utf-8'))
            while not runtime.terminal:
                before = None
                controller_before = taps.controller
                if instrumented:
                    check = signature(runtime)
                    before = snapshot(runtime, blobs)
                    if signature(runtime) != check:
                        raise ValueError('snapshot modified RNG, planner cache, state, or guidance')
                    count += 1
                prior_calls = taps.physics_calls
                taps.controller, taps.guidance = None, None
                begin = time.perf_counter()
                transition = map_contract.advance(runtime, ZeroResidualPolicy(), explore=False)
                elapsed.append(time.perf_counter()-begin)
                physical_steps += 1
                signatures.append(signature(runtime, transition))
                if instrumented:
                    if taps.physics_calls != prior_calls+1:
                        raise ValueError('each physical step must forward exactly one dynamics call')
                    frame = dict(schema=SCHEMA, step_before=int(transition['t']), before=before,
                        physical_step=taps.physics, transition=plain(transition),
                        controller_before_step=controller_before,
                        controller_after_step=taps.controller,
                        compiled_after_step=taps.guidance)
                    stream.write(json.dumps(frame, allow_nan=False)+'\n')
                    stream.flush()
            summary = plain(runtime.summary())
            write(output/'episode.json', summary)
    except Exception as error:
        write(output/'failure.json', dict(completed=False, error=repr(error),
            physical_steps=physical_steps, completed_snapshots=count))
        raise
    finally:
        if runtime is not None:
            runtime.close()
        torch.set_num_threads(previous)
    write(output/'signatures.json', signatures)
    inventory = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'recording.json', dict(schema=SCHEMA, completed=True, instrumented=instrumented,
        physical_steps=physical_steps, snapshot_checks=count, forwarded_physics_calls=taps.physics_calls,
        files=inventory, wall_seconds=sum(elapsed), method='D2 zero normalized residual',
        trace_timing_scope='advance includes observer taps only in the recorded arm; not a policy performance comparison',
        ground_truth_geometry_used_during_capture=False, checkpoint_restore=False))
    return summary, signatures


def verified_recording(directory):
    directory = Path(directory)
    receipt = read(directory/'recording.json')
    if receipt.get('schema') != SCHEMA or not receipt.get('completed'):
        raise ValueError('incomplete diagnostic recording')
    actual = {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob('*')
              if p.is_file() and p.name != 'recording.json'}
    if receipt['files'] != actual:
        raise ValueError('diagnostic recording inventory changed')
    summary=read(directory/'episode.json');signatures=read(directory/'signatures.json')
    count=receipt['physical_steps']
    if count!=summary['actual_length'] or len(signatures)!=count+1:
        raise ValueError('recording clock/signature count mismatch')
    if any(not isinstance(s,str) or len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in signatures):
        raise ValueError('invalid behavioral signature')
    frames=[json.loads(line) for line in (directory/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    if receipt['instrumented']:
        if len(frames)!=count or receipt['forwarded_physics_calls']!=count or receipt['snapshot_checks']!=count:
            raise ValueError('incomplete recorded physical steps')
        if any(f['schema']!=SCHEMA or f['step_before']!=i or f['transition']['t']!=i for i,f in enumerate(frames)):
            raise ValueError('recorded step order changed')
        hits=[(i,f) for i,f in enumerate(frames) if f['physical_step']['collisions']]
        if hits:
            i,frame=hits[0]
            if (len(hits)!=1 or i!=count-1 or summary['first_collision_step']!=count
                or summary['collision_records']!=frame['physical_step']['collisions']
                or summary['first_collision_phase']!=frame['physical_step']['before']['phase']):
                raise ValueError('collision summary differs from physical event')
        elif summary.get('collision_records'):
            raise ValueError('collision summary has no physical event')
    elif frames:
        raise ValueError('plain arm unexpectedly contains observer frames')
    return receipt
