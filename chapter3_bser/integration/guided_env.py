from __future__ import annotations
from dataclasses import replace
from types import MethodType
from typing import Any
from typing import Mapping
import torch
from chapter3_bser.integration.control_context import BSERControlContextV1
from chapter3_bser.integration.control_context import Vector3
from chapter3_bser.integration.rmaddpg_bridge import get_tracking_targets

class GuidedEnv:
    """Install BSER waypoints while preserving the environment action contract.

    The wrapper is inactive unless constructed with ``enabled=True``.  When
    active, it lets the underlying environment perform its normal navigation
    update and then reinstalls the cached BSER tracking targets before the
    waypoint prior or observation is computed.  No velocity or action is
    generated here.
    """

    def __init__(self, env: Any, *, enabled: bool=False) -> None:
        self.env = env
        self.enabled = bool(enabled)
        self._runtime = getattr(env, 'unwrapped', env)
        self._active_targets: tuple[Vector3, ...] | None = None
        self._public_executor_navigation_target: Vector3 | None = None
        self._installed_context: BSERControlContextV1 | None = None
        self._original_update_nav_targets = None
        self._initial_endpoint_fallback_count = 0
        self._last_initial_endpoint_failures: tuple[dict[str, Any], ...] = ()
        self._closed = False
        if self.enabled:
            self._install_navigation_hook()

    def _install_navigation_hook(self) -> None:
        update = getattr(self._runtime, '_update_nav_targets', None)
        if not callable(update):
            raise TypeError('guided environment requires a navigation-target update hook')
        self._original_update_nav_targets = update
        owner = self

        def _guided_update(runtime_self, *args, **kwargs):
            result = owner._original_update_nav_targets(*args, **kwargs)
            owner._cache_public_executor_navigation_target()
            owner._apply_navigation_targets()
            return result
        self._runtime._update_nav_targets = MethodType(_guided_update, self._runtime)

    def _executor_id(self) -> int:
        return int(getattr(self._runtime, 'executor_id', 3))

    def _cache_public_executor_navigation_target(self) -> None:
        targets = getattr(self._runtime, '_nav_targets', None)
        if targets is None:
            return
        try:
            value = targets[self._executor_id()]
        except (IndexError, KeyError, TypeError):
            return
        if torch.is_tensor(value):
            values = value.detach().cpu().reshape(3).tolist()
        else:
            values = list(value)
        if len(values) != 3:
            raise ValueError('public executor navigation target must be three-dimensional')
        self._public_executor_navigation_target = tuple((float(item) for item in values))

    @property
    def public_executor_navigation_target(self) -> Vector3 | None:
        value = self._public_executor_navigation_target
        return None if value is None else tuple((float(item) for item in value))

    def get_public_executor_navigation_target(self) -> Vector3 | None:
        return self.public_executor_navigation_target

    def get_agent_state(self):
        public = self.env.get_agent_state()
        target = self._public_executor_navigation_target
        if target is None:
            return public
        task = self.env.get_task_state()
        if not bool(task.executor_knows_target):
            return public
        executor_id = self._executor_id()
        targets = list(public.navigation_targets)
        targets[executor_id] = tuple((float(item) for item in target))
        return replace(public, navigation_targets=tuple(targets))

    def _apply_navigation_targets(self) -> None:
        if self._active_targets is None:
            return
        targets = torch.as_tensor(self._active_targets, dtype=self._runtime._nav_targets.dtype, device=self._runtime._nav_targets.device)
        if tuple(targets.shape) != tuple(self._runtime._nav_targets.shape):
            raise ValueError('guidance target shape does not match environment navigation targets')
        with torch.no_grad():
            self._runtime._nav_targets.copy_(targets)
            if hasattr(self._runtime, '_targets'):
                self._runtime._targets.copy_(targets)

    @property
    def installed_context(self) -> BSERControlContextV1 | None:
        return self._installed_context

    @property
    def installed_allocation_version(self) -> str | None:
        return None if self._installed_context is None else self._installed_context.allocation_version

    @property
    def initial_endpoint_fallback_count(self) -> int:
        """Number of resets protected from an invalid online-map endpoint."""
        return int(self._initial_endpoint_fallback_count)

    def install_guidance(self, context: BSERControlContextV1) -> None:
        """Install navigation intent without advancing environment state."""
        if not self.enabled:
            raise RuntimeError('Phase 1C BSER guidance is disabled')
        if not isinstance(context, BSERControlContextV1):
            raise TypeError('install_guidance requires BSERControlContextV1')
        targets: Mapping[int, Vector3] = get_tracking_targets(context)
        expected = tuple(range(int(self._runtime.num_agents)))
        if tuple(sorted(targets)) != expected:
            raise ValueError('guidance must contain exactly one target per agent')
        self._active_targets = tuple((targets[index] for index in expected))
        self._installed_context = context
        self._apply_navigation_targets()

    def refresh_observation_after_guidance(self):
        """Recompute only the 28D observation after a guidance change."""
        if not self.enabled or self._installed_context is None:
            raise RuntimeError('guidance must be installed before refreshing observation')
        self._apply_navigation_targets()
        observations = self._runtime._obs_to_public(self._runtime._get_obs())
        if hasattr(self.env, '_last_observations'):
            self.env._last_observations = observations
        return observations

    def reset(self, scenario=None):
        self._active_targets = None
        self._public_executor_navigation_target = None
        self._installed_context = None
        self._last_initial_endpoint_failures = ()
        if not self.enabled or not bool(getattr(self._runtime, 'online_unknown_map_active', False)):
            observations = self.env.reset(scenario=scenario)
            if self._public_executor_navigation_target is None:
                self._cache_public_executor_navigation_target()
            return observations
        planner = getattr(self._runtime, 'map_module', None)
        initial_targets = getattr(planner, 'initial_search_targets', None)
        endpoint_status = getattr(planner, 'endpoint_status', None)
        if not callable(initial_targets) or not callable(endpoint_status):
            observations = self.env.reset(scenario=scenario)
            if self._public_executor_navigation_target is None:
                self._cache_public_executor_navigation_target()
            return observations
        had_instance_override = 'initial_search_targets' in planner.__dict__
        previous_override = planner.__dict__.get('initial_search_targets')
        fallback_used = False

        def guarded_initial_targets(agent_positions):
            nonlocal fallback_used
            failures = []
            positions = planner._as_points(agent_positions).reshape(-1, 3)
            for (agent_id, position) in enumerate(positions):
                status = endpoint_status(position, role='searcher')
                if status.get('reachable'):
                    continue
                failures.append({'agent_id': int(agent_id), 'failure_reason': str(status.get('failure_reason')), 'position': tuple((float(value) for value in position.detach().cpu().tolist()))})
            if not failures:
                return initial_targets(agent_positions)
            if not fallback_used:
                self._initial_endpoint_fallback_count += 1
                fallback_used = True
            self._last_initial_endpoint_failures = tuple(failures)
            return positions.detach().clone()
        planner.initial_search_targets = guarded_initial_targets
        try:
            observations = self.env.reset(scenario=scenario)
        finally:
            if had_instance_override:
                planner.initial_search_targets = previous_override
            else:
                del planner.initial_search_targets
        if self._public_executor_navigation_target is None:
            self._cache_public_executor_navigation_target()
        return observations

    def step(self, actions):
        return self.env.step(actions)

    def close(self) -> None:
        if self._closed:
            return
        if self._original_update_nav_targets is not None:
            self._runtime._update_nav_targets = self._original_update_nav_targets
        self._closed = True
        close = getattr(self.env, 'close', None)
        if callable(close):
            close()

    def __getattr__(self, name: str):
        return getattr(self.env, name)
