from __future__ import annotations
import inspect
from typing import Any
from typing import Dict
from typing import Tuple
import numpy as np
import torch
from core.env.observation_contract import OBSERVATION_DIM
from core.env.observation_contract import ROLE_ORDER
from core.env.task_state import AgentStateView
from core.env.task_state import MissionTaskState
from core.env.task_state import SearchExecutionState
from core.env.uav_env import UAVEnv

def environment_kwargs_from_config(config: Dict[str, Any], *, device: str='cpu', max_steps: int=50, return_numpy: bool=False) -> Dict[str, Any]:
    """Apply the exact constructor-key filtering used by CH3/train.py."""
    constructors = (UAVEnv.__mro__[1].__init__, UAVEnv.__init__)
    valid = {name for constructor in constructors for (name, parameter) in inspect.signature(constructor).parameters.items() if name not in {'self', 'kwargs'} and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}}
    for constructor in constructors:
        code = constructor.__code__
        positional = code.co_argcount
        keyword_only = code.co_kwonlyargcount
        valid.update(code.co_varnames[:positional + keyword_only])
    valid.discard('self')
    kwargs = {key: value for (key, value) in dict(config).items() if key in valid}
    kwargs.update(device=device, return_numpy=bool(return_numpy), max_steps=int(max_steps))
    return kwargs

def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()

def _vectors(value: Any) -> Tuple[Tuple[float, float, float], ...]:
    rows = _array(value).reshape(-1, 3)
    return tuple((tuple((float(x) for x in row)) for row in rows))

class MissionCoreEnv:
    """Compatibility-first canonical environment API.

    All dynamics, reward, task events, mapping, target motion, and observations
    are delegated without modification to the self-contained core implementation.
    """
    n_agents = 4
    role_order = ROLE_ORDER
    local_observation_dims = (28, 28, 28, 28)
    action_dims = (3, 3, 3, 3)
    reward_shape = (4,)
    action_type = 'continuous residual acceleration'
    action_range = (-1.0, 1.0)
    communication_mode = 'ch3_fixed_reliable'
    implementation_source = 'core'

    def __init__(self, **kwargs: Any) -> None:
        if int(kwargs.get('n_agents', 4)) != self.n_agents:
            raise ValueError('MissionCoreEnv requires exactly four agents')
        device = torch.device(kwargs.get('device') or 'cpu')
        if device.type == 'cpu':
            kwargs['device'] = 'cpu'
        self._env = UAVEnv(**kwargs)
        self.return_numpy = bool(self._env.return_numpy)
        self._last_observations = None
        if int(self._env.obs_dim) != OBSERVATION_DIM:
            raise RuntimeError('core observation dimension is not 28')

    @property
    def unwrapped(self) -> UAVEnv:
        """Read-only access for equivalence instrumentation only."""
        return self._env

    def reset(self, scenario=None):
        self._last_observations = self._env.reset(scenario=scenario)
        return self._last_observations

    def step(self, actions):
        result = self._env.step(actions)
        self._last_observations = result[0]
        return result

    def get_task_state(self) -> MissionTaskState:
        return MissionTaskState(step=int(self._env.step_count), target_found=bool(self._env.task_found), finder_id=int(self._env.finder_idx), executor_knows_target=bool(self._env.executor_target_assigned), handoff_step=None if self._env.handoff_step is None else int(self._env.handoff_step), mission_complete=bool(self._env.mission_complete), completion_step=None if self._env.success_step is None else int(self._env.success_step))

    def get_episode_result(self) -> Dict[str, Any]:
        """Authoritative termination metadata, separate from the frozen 28D input."""
        return self._env.get_episode_result()

    def get_agent_state(self) -> AgentStateView:
        return AgentStateView(role_order=self.role_order, positions=_vectors(self._env._agent_pos), velocities=_vectors(self._env._agent_vel), navigation_targets=_vectors(self._env._nav_targets), collision_flags=tuple((bool(x) for x in _array(self._env._collision_flags).reshape(-1))))

    def get_search_execution_state(self) -> SearchExecutionState:
        return SearchExecutionState(searcher_ids=(0, 1, 2), executor_id=3, target_known_by_agent=tuple((bool(x) for x in _array(self._env._agent_task_known).reshape(-1))), waypoint_reached_counts=tuple((int(x) for x in _array(self._env.waypoint_reached_counts).reshape(-1))), agent_finished=tuple((bool(x) for x in _array(self._env.agent_finished).reshape(-1))), hold_counters=tuple((int(x) for x in _array(self._env.hold_counters).reshape(-1))))

    def get_scenario_identity(self) -> Dict[str, Any]:
        return {'scenario_profile': str(self._env.scenario_profile), 'obstacle_layout_id': str(self._env.obstacle_layout_id), 'obstacle_knowledge_mode': str(self._env.obstacle_knowledge_mode), 'base_candidate': str(self._env.base_candidate)}

    def close(self) -> None:
        close = getattr(self._env, 'close', None)
        if callable(close):
            close()

    def enable_public_event_capture(self, episode_id, *, capacity=1024):
        if not self._env.online_unknown_map_active:
            raise ValueError('public capture currently supports the online unknown-map mission')
        self._env.enable_public_event_capture(episode_id, capacity=capacity)

    def read_public_events(self, after_sequence=0):
        return self._env.read_public_events(after_sequence)

    def get_public_vehicle_specs(self):
        fields = ('name', 'a_xy_max', 'a_z_max', 'v_xy_max', 'v_z_max',
                  'drag_xy', 'drag_z', 'buoyancy_bias', 'sensor_range')
        return tuple({key: spec[key] for key in fields} for spec in self._env.agent_specs)

    def get_public_control_contract(self):
        fields = ('dt', 'prior_kv_xy', 'prior_kv_z', 'prior_slow_radius_xy',
                  'prior_slow_radius_z', 'prior_strength_search', 'prior_strength_executor',
                  'residual_scale_search', 'residual_scale_executor', 'detect_eps_bias',
                  'target_motion_mode', 'target_obstacle_clearance', 'target_max_reflections_per_step',
                  'obstacle_sensor_range', 'use_residual_prior', 'eps')
        result = {key: getattr(self._env, key) for key in fields}
        result['bounds'] = tuple(float(x) for x in _array(self._env.space_size))
        return result
