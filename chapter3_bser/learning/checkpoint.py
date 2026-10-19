from __future__ import annotations
from project_paths import ROOT as _PROJECT_ROOT
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import torch
from chapter3_bser.rewards.team_objective import TEAM
from chapter3_bser.rewards.team_objective import objective_identity
from core.env.task_protocol import STRICT
from core.env.task_protocol import protocol_identity
from core.env.task_protocol import require_protocol_output
from core.env.task_protocol import validate_task_config
ROOT = _PROJECT_ROOT
CONFIG_SCHEMA = 'ch3.baseline.training.v1'
CHECKPOINT_SCHEMA = 'ch3.baseline.maddpg.v1'
IMPLEMENTATION_VERSION = 'auv.active_source.rebuild.v1'
IDENTITIES = {'B2_direct_mc': ('maddpg', 'ch3_baseline_direct_mc', 'ch3.baseline.maddpg.v1'), 'B3_direct_boundary': ('direct_boundary_maddpg', 'ch3_baseline_direct_boundary', 'ch3.baseline.boundary_maddpg.v1')}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def validate_config(config, expected_baseline=None):
    config = copy.deepcopy(dict(config))
    from chapter3_bser.d2.contract import planner_protocol
    planner_protocol(config)
    from chapter3_bser.d2.options import performance_options
    performance_options(config)
    if config.get('schema') != CONFIG_SCHEMA:
        raise ValueError('not an independent baseline training config; an HGR checkpoint cannot be relabeled')
    baseline = config.get('baseline')
    if baseline not in IDENTITIES or (expected_baseline is not None and baseline != expected_baseline):
        raise ValueError('baseline identity mismatch')
    (algorithm, method, architecture) = IDENTITIES[baseline]
    for (key, expected) in (('algorithm', algorithm), ('method', method), ('architecture_version', architecture), ('checkpoint_schema', CHECKPOINT_SCHEMA)):
        if config.get(key) != expected:
            raise ValueError(f'baseline {key} mismatch; relabeling weights is forbidden')
    forbidden = {'policy', 'predictor', 'prefix_lr', 'suffix_lr', 'main_prefix_batch_size', 'suffix_training_episodes_per_cycle', 'pilot_prefix_episodes_per_cycle', 'correction_draws_per_cycle'}
    if forbidden.intersection(config):
        raise ValueError('independent MADDPG baselines do not accept HGR cycle/policy/predictor fields')
    validate_task_config(config)
    if protocol_identity(config)['task_protocol'] != STRICT or objective_identity(config)['reward_objective'] != TEAM:
        raise ValueError('baselines require collision_terminal_v1 and team_mean_v1')
    if config.get('observation_dim') != 28 or config.get('action_dim') != 3 or config.get('critic_input_dim', 124) != 124:
        raise ValueError('baseline 28D observation/3D action/124D critic contract mismatch')
    if config.get('base_candidate') != 'ch3_v3_full_reference' or config.get('profile') != 'M20_MOVING_UNKNOWN_MULTI':
        raise ValueError('baseline environment/profile mismatch')
    if config.get('ablation', 'none') != 'none' or config.get('early_discovery', {}).get('enabled'):
        raise ValueError('baseline training does not accept ablations')
    for key in ('search_value', 'search_value_decision', 'search_value_guidance'):
        if config.get(key, {}).get('enabled'):
            raise ValueError('baseline training does not use historical search-value guidance')
    reward = config.get('reward', {})
    if not reward.get('freeze_searchers_after_found') or reward.get('executor_id') != 3 or reward.get('searcher_ids') != [0, 1, 2]:
        raise ValueError('baseline reward/frozen searcher/role contract mismatch')
    for key in ('total_main_trajectories', 'checkpoint_interval', 'max_steps'):
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if isinstance(config.get('seed'), bool) or not isinstance(config.get('seed'), int) or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    budget = config.get('max_total_environment_steps')
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
        raise ValueError('max_total_environment_steps must be null or a positive integer')
    rl = config.get('rl', {})
    for key in ('batch_size', 'replay_size', 'update_frequency', 'updates_per_train', 'hidden_dim'):
        if isinstance(rl.get(key), bool) or not isinstance(rl.get(key), int) or rl[key] < 1:
            raise ValueError(f'rl.{key} must be a positive integer')
    if rl['replay_size'] < rl['batch_size']:
        raise ValueError('replay capacity must be at least batch_size')
    if isinstance(rl.get('warmup_steps'), bool) or not isinstance(rl.get('warmup_steps'), int) or rl['warmup_steps'] < 0:
        raise ValueError('rl.warmup_steps must be a nonnegative integer')
    for key in ('gamma', 'tau', 'lr_actor', 'lr_critic'):
        if not isinstance(rl.get(key), (int, float)) or not math.isfinite(rl[key]) or rl[key] <= 0:
            raise ValueError(f'rl.{key} must be finite and positive')
    if rl['gamma'] > 1 or rl['tau'] > 1 or rl.get('residual_action_reg') != 0:
        raise ValueError('invalid gamma/tau or nonzero expert residual regularization')
    if baseline == 'B2_direct_mc' and 'boundary_encoder' in config:
        raise ValueError('B2 has no boundary encoder')
    if baseline == 'B3_direct_boundary':
        boundary = config.get('boundary_encoder', {})
        if boundary.get('schema', 'ch3.baseline.boundary_encoder.v1') != 'ch3.baseline.boundary_encoder.v1' or type(boundary.get('hidden_dim', 32)) is not int or boundary.get('hidden_dim', 32) < 1:
            raise ValueError('invalid boundary encoder schema or hidden dimension')
    config.update(objective_identity(config))
    return config

def validated_output(config, output):
    path = Path(output).resolve()
    require_protocol_output(config, path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'use a new or empty output directory: {path}')
    return path

def fresh_source_identity():
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for package in ('core', 'chapter3_bser') for p in sorted((ROOT / package).rglob('*.py'))}
    return dict(implementation_version=IMPLEMENTATION_VERSION, sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=files)

def validate_source_identity(identity):
    if not isinstance(identity, dict) or identity.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('invalid baseline production source identity version')
    files = identity.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('production source identity requires its complete file inventory')
    for (name, value) in files.items():
        if not isinstance(name, str):
            raise ValueError('invalid production source filename')
        path = PurePosixPath(name)
        if name != path.as_posix() or path.is_absolute() or (not path.parts) or ('..' in path.parts) or (path.parts[0] not in ('core', 'chapter3_bser')) or (path.suffix != '.py') or (not isinstance(value, str)) or (re.fullmatch('[0-9a-f]{64}', value) is None):
            raise ValueError(f'invalid production source record: {name!r}')
    if identity.get('sha256') != hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest():
        raise ValueError('production source inventory/aggregate hash mismatch')
    return identity

def require_source_match(saved, current=None, *, context='evaluation'):
    validate_source_identity(saved)
    current = fresh_source_identity() if current is None else validate_source_identity(current)
    if saved != current:
        changed = sorted((name for name in saved['files'].keys() | current['files'].keys() if saved['files'].get(name) != current['files'].get(name)))
        raise ValueError(f'{context} production source mismatch: ' + ', '.join(changed[:12]))

def state_digest(state):
    """Hash tensors and optimizer state without serialization-container timestamps."""
    hashed = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            hashed.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            hashed.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                hashed.update(repr(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            hashed.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        else:
            hashed.update(repr(value).encode())
    visit(state)
    return hashed.hexdigest()

def _finite_state(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all((_finite_state(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return all((_finite_state(item) for item in value))
    return not isinstance(value, float) or math.isfinite(value)
