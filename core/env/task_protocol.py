from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import math
import numpy as np
LEGACY = 'legacy_nonterminal_v1'
STRICT = 'collision_terminal_v1'
PROTOCOL_FIELDS = ('task_protocol', 'collision_detection_revision', 'terminal_reward_revision')
REVISIONS = {LEGACY: ('endpoint_rollback_v1', 'legacy_shaping_v1'), STRICT: ('segment_closed_aabb_v1', 'team_failure_override_v1')}
COLLISION_EPS = 1e-09

def protocol_identity(config=None):
    config = {} if config is None else config
    protocol = config.get('task_protocol', LEGACY)
    if protocol not in REVISIONS:
        raise ValueError(f'unsupported task_protocol: {protocol!r}')
    (detection, reward) = REVISIONS[protocol]
    identity = dict(zip(PROTOCOL_FIELDS, (protocol, detection, reward)))
    for (key, expected) in identity.items():
        if key in config and config[key] != expected:
            raise ValueError(f'incompatible {key}: {config[key]!r}; expected {expected!r}')
    return identity

def validate_task_config(config):
    identity = protocol_identity(config)
    if identity['task_protocol'] == STRICT:
        value = float(config.get('collision_terminal_reward', -2.0))
        if not math.isfinite(value) or value >= 0:
            raise ValueError('collision_terminal_reward must be finite and strictly negative')
        reward = config.get('reward')
        if reward is None:
            return identity
        clip = float(reward.get('reward_clip', reward.get('reward_clip_abs', 3.0)))
        if reward.get('enabled', True) and (not math.isfinite(clip) or (clip > 0 and clip < abs(value))):
            raise ValueError('reward_clip must preserve collision_terminal_reward exactly')
    return identity

def require_protocol_output(config, output):
    from pathlib import Path
    if protocol_identity(config)['task_protocol'] == STRICT and 'collision_terminal' not in Path(output).parts:
        raise ValueError('strict protocol output must use an independent collision_terminal directory')

def closed_segment_aabb_first_hit(start, end, lower, upper):
    """Closed point-segment / real AABB slab test, including stationary contact.

    The target-reflection helper excludes stationary/interior starts to avoid
    repeated reflections. Physics therefore uses this separate closed-set test.
    Bounds expand by only 1e-9 world units; returned tau is clamped to [0, 1].
    """
    (start, end, lower, upper) = [np.asarray(v, dtype=np.float64) for v in (start, end, lower, upper)]
    if any((v.shape != (3,) or not np.isfinite(v).all() for v in (start, end, lower, upper))):
        raise ValueError('collision geometry requires finite 3-vectors')
    if np.any(lower > upper):
        raise ValueError('invalid AABB bounds')
    (lo, hi) = (lower - COLLISION_EPS, upper + COLLISION_EPS)
    (enter, leave) = (0.0, 1.0)
    for (axis, delta) in enumerate(end - start):
        if delta == 0.0:
            if start[axis] < lo[axis] or start[axis] > hi[axis]:
                return None
        else:
            (a, b) = sorted(((lo[axis] - start[axis]) / delta, (hi[axis] - start[axis]) / delta))
            (enter, leave) = (max(enter, a), min(leave, b))
            if enter > leave:
                return None
    return float(enter)

@dataclass
class EpisodeOutcome:
    episode_terminated: bool = False
    episode_truncated: bool = False
    termination_reason: str = 'running'
    terminal_step: int | None = None
    first_collision_step: int | None = None
    first_collision_agent_ids: list = field(default_factory=list)
    first_collision_phase: str | None = None
    collision_records: list = field(default_factory=list)
    success: bool = False
    mission_complete: bool = False

    def finish(self, reason, step):
        if self.episode_terminated:
            return
        self.episode_terminated = True
        self.termination_reason = reason
        self.terminal_step = int(step)
        self.success = self.mission_complete = reason == 'success'

    def to_dict(self):
        return asdict(self)

def strict_terminal(env):
    runtime = getattr(env, 'unwrapped', env)
    return getattr(runtime, 'task_protocol', LEGACY) == STRICT and runtime.episode_outcome.episode_terminated

def episode_result(runtime):
    """Copy the authoritative outcome for collectors and evaluators."""
    result = runtime.episode_outcome.to_dict()
    result.update(protocol_identity(vars(runtime)))
    result.update(terminated=result['episode_terminated'], truncated=result['episode_truncated'], safe_success=bool(result['success'] and (not result['collision_records'])), collision_episode=bool(result['collision_records']), remaining_task_steps=max(0, runtime.max_steps - runtime.step_count), bootstrap_mask=0.0 if result['episode_terminated'] else 1.0)
    return result
