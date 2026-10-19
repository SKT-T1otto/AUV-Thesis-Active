from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any
from typing import Mapping
import numpy as np
import torch

def _runtime(env: Any) -> Any:
    return getattr(env, 'unwrapped', getattr(env, 'env', env))

def _number(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)

def _vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != 3 or not np.all(np.isfinite(array)):
        return None
    return array

def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))

def _sum_reward(value: Any) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    return float(torch.nan_to_num(tensor).sum().item())

def _event_names(controller_result: Any) -> tuple[str, ...]:
    if controller_result is None:
        return ()
    events = getattr(controller_result, 'events', ()) or ()
    names: list[str] = []
    for event in events:
        raw = getattr(event, 'value', event)
        names.append(str(raw).upper())
    return tuple(names)

@dataclass
class ExecutionEpisodeDiagnostics:
    """Collect post-discovery execution-chain evidence without mutating runtime."""
    episode_index: int = 0
    episode_id: int = 0
    scenario_id: str | None = None
    scenario_seed: int | None = None
    max_steps: int = 0
    found_step: int | None = None
    success_step: int | None = None
    executor_target_received_step: int | None = None
    executor_distance_to_target_at_found: float | None = None
    executor_distance_to_target_at_received: float | None = None
    executor_distance_to_intercept_at_received: float | None = None
    executor_min_distance_to_target: float | None = None
    executor_final_distance_to_target: float | None = None
    executor_min_distance_to_intercept: float | None = None
    executor_final_distance_to_intercept: float | None = None
    pre_found_base_reward: float = 0.0
    post_found_base_reward: float = 0.0
    post_found_shaping_reward: float = 0.0
    completion_reward_post_tanh: float = 0.0
    adjusted_episode_reward: float = 0.0
    post_found_collision_count: int = 0
    executor_invalid_count: int = 0
    executor_replan_count: int = 0
    executor_invalid_assignment_unreachable_count: int = 0
    executor_invalid_query_unreachable_count: int = 0
    executor_invalid_cost_increase_count: int = 0
    executor_invalid_stale_snapshot_deferred_count: int = 0
    executor_validity_evaluation_count: int = 0
    executor_validity_deferred_count: int = 0
    public_target_update_event_count: int = 0
    public_target_update_accepted_count: int = 0
    environment_public_target_at_handoff: tuple[float, float, float] | None = None
    controller_execution_target_at_handoff: tuple[float, float, float] | None = None
    installed_executor_tracking_target_at_handoff: tuple[float, float, float] | None = None
    controller_to_public_target_error_at_handoff: float | None = None
    full_planning_refresh_count: int = 0
    handoff_forced_refresh_count: int = 0
    target_shift_forced_refresh_count: int = 0
    _public_target_shifts: list[float] = field(default_factory=list)
    _controller_to_public_target_errors: list[float] = field(default_factory=list)
    _search_residual_pre: list[float] = field(default_factory=list)
    _search_residual_post: list[float] = field(default_factory=list)
    _executor_residual_pre: list[float] = field(default_factory=list)
    _executor_residual_post: list[float] = field(default_factory=list)
    _last_task_found: bool = False

    def reset(self, env: Any, *, episode_index: int, episode_id: int | None=None, scenario_id: str | None=None, scenario_seed: int | None=None, max_steps: int | None=None) -> 'ExecutionEpisodeDiagnostics':
        self.episode_index = int(episode_index)
        self.episode_id = int(episode_index if episode_id is None else episode_id)
        self.scenario_id = None if scenario_id is None else str(scenario_id)
        self.scenario_seed = None if scenario_seed is None else int(scenario_seed)
        runtime = _runtime(env)
        self.max_steps = int(max_steps if max_steps is not None else getattr(runtime, 'max_steps', 0))
        self.found_step = None
        self.success_step = None
        self.executor_target_received_step = None
        self.executor_distance_to_target_at_found = None
        self.executor_distance_to_target_at_received = None
        self.executor_distance_to_intercept_at_received = None
        self.executor_min_distance_to_target = None
        self.executor_final_distance_to_target = None
        self.executor_min_distance_to_intercept = None
        self.executor_final_distance_to_intercept = None
        self.pre_found_base_reward = 0.0
        self.post_found_base_reward = 0.0
        self.post_found_shaping_reward = 0.0
        self.completion_reward_post_tanh = 0.0
        self.adjusted_episode_reward = 0.0
        self.post_found_collision_count = 0
        self.executor_invalid_count = 0
        self.executor_replan_count = 0
        self.executor_invalid_assignment_unreachable_count = 0
        self.executor_invalid_query_unreachable_count = 0
        self.executor_invalid_cost_increase_count = 0
        self.executor_invalid_stale_snapshot_deferred_count = 0
        self.executor_validity_evaluation_count = 0
        self.executor_validity_deferred_count = 0
        self.public_target_update_event_count = 0
        self.public_target_update_accepted_count = 0
        self.environment_public_target_at_handoff = None
        self.controller_execution_target_at_handoff = None
        self.installed_executor_tracking_target_at_handoff = None
        self.controller_to_public_target_error_at_handoff = None
        self.full_planning_refresh_count = 0
        self.handoff_forced_refresh_count = 0
        self.target_shift_forced_refresh_count = 0
        self._public_target_shifts = []
        self._controller_to_public_target_errors = []
        self._search_residual_pre = []
        self._search_residual_post = []
        self._executor_residual_pre = []
        self._executor_residual_post = []
        task = env.get_task_state() if hasattr(env, 'get_task_state') else None
        self._last_task_found = bool(getattr(task, 'target_found', False))
        return self

    @staticmethod
    def _distances(env: Any) -> tuple[float | None, float | None]:
        runtime = _runtime(env)
        positions = getattr(runtime, '_agent_pos', None)
        if positions is None:
            positions = getattr(runtime, 'agent_pos', None)
        try:
            executor = _vector(positions[3])
        except (TypeError, IndexError):
            executor = None
        target_state = getattr(runtime, 'target_state', None)
        target = _vector(getattr(target_state, 'position', None))
        intercept = _vector(getattr(runtime, 'predicted_intercept_position', None))
        target_distance = None if executor is None or target is None else float(np.linalg.norm(executor - target))
        intercept_distance = None if executor is None or intercept is None else float(np.linalg.norm(executor - intercept))
        return (target_distance, intercept_distance)

    @staticmethod
    def _exact_executor_unreachable(runtime: Any) -> int | None:
        for name in ('executor_path_unreachable_count', 'path_unreachable_count_executor'):
            if hasattr(runtime, name):
                return _integer(getattr(runtime, name))
        return None

    def observe_step(self, env: Any, rewards: Any, *, task_before: Any, task_after: Any, base_rewards: Any | None=None, reward_breakdown: Mapping[str, Any] | None=None, controller_result: Any | None=None) -> None:
        runtime = _runtime(env)
        found_before = bool(getattr(task_before, 'target_found', False))
        found_after = bool(getattr(task_after, 'target_found', False))
        success_after = bool(getattr(task_after, 'mission_complete', False))
        step = int(getattr(task_after, 'step', 0))
        (target_distance, intercept_distance) = self._distances(env)
        if found_after and self.found_step is None:
            self.found_step = step
            self.executor_distance_to_target_at_found = target_distance
        if success_after and self.success_step is None:
            self.success_step = step
        received = _integer(getattr(runtime, 'executor_received_target_step', None))
        if received is None and bool(getattr(task_after, 'executor_knows_target', False)):
            received = step
        if received is not None and self.executor_target_received_step is None:
            self.executor_target_received_step = received
            self.executor_distance_to_target_at_received = target_distance
            self.executor_distance_to_intercept_at_received = intercept_distance
        if found_after:
            if target_distance is not None:
                self.executor_final_distance_to_target = target_distance
                if self.executor_min_distance_to_target is None:
                    self.executor_min_distance_to_target = target_distance
                else:
                    self.executor_min_distance_to_target = min(self.executor_min_distance_to_target, target_distance)
            if intercept_distance is not None:
                self.executor_final_distance_to_intercept = intercept_distance
                if self.executor_min_distance_to_intercept is None:
                    self.executor_min_distance_to_intercept = intercept_distance
                else:
                    self.executor_min_distance_to_intercept = min(self.executor_min_distance_to_intercept, intercept_distance)
            flags = getattr(runtime, '_collision_flags', ())
            try:
                self.post_found_collision_count += int(torch.as_tensor(flags, dtype=torch.bool).sum().item())
            except (TypeError, ValueError, RuntimeError):
                pass
        search_ratio = _number(getattr(runtime, 'last_residual_contribution_ratio_search', None))
        executor_ratio = _number(getattr(runtime, 'last_residual_contribution_ratio_executor', None))
        if found_after:
            if search_ratio is not None:
                self._search_residual_post.append(search_ratio)
            if executor_ratio is not None:
                self._executor_residual_post.append(executor_ratio)
        else:
            if search_ratio is not None:
                self._search_residual_pre.append(search_ratio)
            if executor_ratio is not None:
                self._executor_residual_pre.append(executor_ratio)
        base_total = _sum_reward(rewards if base_rewards is None else base_rewards)
        if found_before or found_after:
            self.post_found_base_reward += base_total
        else:
            self.pre_found_base_reward += base_total
        self.adjusted_episode_reward += _sum_reward(rewards)
        if reward_breakdown:
            shaping = _number(reward_breakdown.get('execution_shaping_total'))
            if shaping is not None:
                self.post_found_shaping_reward += shaping
            terminal = _number(reward_breakdown.get('terminal_success_bonus_post_tanh'))
            if terminal is None:
                terminal = _number(reward_breakdown.get('terminal_success_bonus'))
            if terminal is not None:
                self.completion_reward_post_tanh += terminal
        if controller_result is not None:
            self.observe_controller_result(env, controller_result)
        self._last_task_found = found_after

    @staticmethod
    def _public_executor_target(env: Any) -> np.ndarray | None:
        getter = getattr(env, 'get_public_executor_navigation_target', None)
        return _vector(getter()) if callable(getter) else None

    @staticmethod
    def _installed_executor_target(env: Any) -> np.ndarray | None:
        runtime = _runtime(env)
        targets = getattr(runtime, '_nav_targets', None)
        try:
            return _vector(targets[3])
        except (TypeError, IndexError):
            return None

    def observe_controller_result(self, env: Any, controller_result: Any, *, controller: Any | None=None, state_provider: Any | None=None) -> None:
        if controller_result is None:
            return
        names = _event_names(controller_result)
        self.executor_invalid_count += sum((item == 'EXECUTOR_INVALID' for item in names))
        if bool(getattr(controller_result, 'replanned', False)):
            diagnostics = getattr(controller_result, 'diagnostics', None)
            affected = tuple(getattr(diagnostics, 'affected_agent_ids', ()) or ())
            scope = str(getattr(diagnostics, 'allocation_scope', ''))
            if 3 in affected or 'executor' in scope:
                self.executor_replan_count += 1
        detection = getattr(controller_result, 'event_detection', None)
        reason = str(getattr(detection, 'executor_invalid_reason', ''))
        if reason == 'ASSIGNMENT_UNREACHABLE':
            self.executor_invalid_assignment_unreachable_count += 1
        elif reason == 'QUERY_UNREACHABLE':
            self.executor_invalid_query_unreachable_count += 1
        elif reason == 'PLANNING_COST_INCREASE':
            self.executor_invalid_cost_increase_count += 1
        elif reason == 'STALE_ENDPOINT_SNAPSHOT_DEFERRED':
            self.executor_invalid_stale_snapshot_deferred_count += 1
        if bool(getattr(detection, 'executor_validity_evaluated', False)):
            self.executor_validity_evaluation_count += 1
        if bool(getattr(detection, 'executor_validity_deferred', False)):
            self.executor_validity_deferred_count += 1
        target_update_event = 'EXECUTOR_PUBLIC_TARGET_UPDATED' in names
        if target_update_event:
            self.public_target_update_event_count += 1
            if bool(getattr(controller_result, 'replanned', False)):
                self.public_target_update_accepted_count += 1
        shift = _number(getattr(detection, 'executor_public_target_shift', None))
        if shift is not None and shift > 0.0:
            self._public_target_shifts.append(shift)
        if state_provider is not None:
            self.full_planning_refresh_count = int(getattr(state_provider, 'full_refresh_count', 0))
            self.handoff_forced_refresh_count = int(getattr(state_provider, 'handoff_forced_refresh_count', 0))
            self.target_shift_forced_refresh_count = int(getattr(state_provider, 'target_shift_forced_refresh_count', 0))
        public_target = self._public_executor_target(env)
        controller_target = _vector(getattr(controller, 'execution_target', None))
        if public_target is not None and controller_target is not None:
            error = float(np.linalg.norm(controller_target - public_target))
            self._controller_to_public_target_errors.append(error)
        else:
            error = None
        if 'EXECUTOR_TARGET_RECEIVED' in names:
            installed = self._installed_executor_target(env)
            self.environment_public_target_at_handoff = None if public_target is None else tuple((float(value) for value in public_target))
            self.controller_execution_target_at_handoff = None if controller_target is None else tuple((float(value) for value in controller_target))
            self.installed_executor_tracking_target_at_handoff = None if installed is None else tuple((float(value) for value in installed))
            self.controller_to_public_target_error_at_handoff = error

    def finalize(self, env: Any) -> dict[str, Any]:
        runtime = _runtime(env)
        task = env.get_task_state() if hasattr(env, 'get_task_state') else None
        found = bool(getattr(task, 'target_found', self.found_step is not None))
        success = bool(getattr(task, 'mission_complete', self.success_step is not None))
        found_step = self.found_step
        if found_step is None:
            found_step = _integer(getattr(runtime, 'found_step', None))
        success_step = self.success_step
        if success_step is None:
            success_step = _integer(getattr(runtime, 'success_step', None))
        received = self.executor_target_received_step
        if received is None:
            received = _integer(getattr(runtime, 'executor_received_target_step', None))
        (final_target, final_intercept) = self._distances(env)
        if found and final_target is not None:
            self.executor_final_distance_to_target = final_target
            if self.executor_min_distance_to_target is None:
                self.executor_min_distance_to_target = final_target
        if found and final_intercept is not None:
            self.executor_final_distance_to_intercept = final_intercept
            if self.executor_min_distance_to_intercept is None:
                self.executor_min_distance_to_intercept = final_intercept
        handoff_delay = _number(getattr(runtime, 'last_handoff_delay', None))
        if handoff_delay is None and found_step is not None and (received is not None):
            handoff_delay = float(received - found_step)
        remaining = None if found_step is None or self.max_steps <= 0 else max(0, int(self.max_steps) - int(found_step))
        initial_fallback = _integer(getattr(env, 'initial_endpoint_fallback_count', None))
        if initial_fallback is None:
            initial_fallback = _integer(getattr(runtime, 'initial_endpoint_fallback_count', None))
        return {'episode_id': int(self.episode_id), 'episode_index': int(self.episode_index), 'scenario_id': self.scenario_id, 'scenario_seed': self.scenario_seed, 'found': found, 'success': success, 'success_if_found': bool(success and found), 'found_step': found_step, 'success_step': success_step, 'remaining_steps_after_found': remaining, 'executor_target_received_step': received, 'handoff_delay': handoff_delay, 'found_to_target_received_steps': None if found_step is None or received is None else int(received - found_step), 'found_to_success_steps': None if found_step is None or success_step is None else int(success_step - found_step), 'executor_distance_at_handoff': self.executor_distance_to_target_at_received, 'executor_distance_to_target_at_found': self.executor_distance_to_target_at_found, 'executor_distance_to_target_at_received': self.executor_distance_to_target_at_received, 'executor_min_distance_to_target': self.executor_min_distance_to_target, 'executor_final_distance_to_target': self.executor_final_distance_to_target, 'executor_distance_to_intercept_at_received': self.executor_distance_to_intercept_at_received, 'executor_min_distance_to_intercept': self.executor_min_distance_to_intercept, 'executor_final_distance_to_intercept': self.executor_final_distance_to_intercept, 'target_prediction_error_at_delivery': _number(getattr(runtime, 'target_prediction_error_at_delivery', None)), 'mean_target_prediction_error': _number(getattr(runtime, 'mean_target_prediction_error', None)), 'target_prediction_map_fallback_count': _integer(getattr(runtime, 'target_prediction_map_fallback_count', None)), 'capture_contact_step_count': _integer(getattr(runtime, 'capture_contact_step_count', None)), 'capture_full_hold_step_count': _integer(getattr(runtime, 'capture_full_hold_step_count', None)), 'capture_hold_counter_max': _integer(getattr(runtime, 'capture_hold_counter_max', None)), 'capture_swept_min_distance': _number(getattr(runtime, 'capture_swept_min_distance', None)), 'executor_path_unreachable_count': self._exact_executor_unreachable(runtime), 'executor_invalid_count': int(self.executor_invalid_count), 'executor_replan_count': int(self.executor_replan_count), 'executor_invalid_assignment_unreachable_count': int(self.executor_invalid_assignment_unreachable_count), 'executor_invalid_query_unreachable_count': int(self.executor_invalid_query_unreachable_count), 'executor_invalid_cost_increase_count': int(self.executor_invalid_cost_increase_count), 'executor_invalid_stale_snapshot_deferred_count': int(self.executor_invalid_stale_snapshot_deferred_count), 'executor_validity_evaluation_count': int(self.executor_validity_evaluation_count), 'executor_validity_deferred_count': int(self.executor_validity_deferred_count), 'public_target_update_event_count': int(self.public_target_update_event_count), 'public_target_update_accepted_count': int(self.public_target_update_accepted_count), 'public_target_shift_sum': float(sum(self._public_target_shifts)), 'public_target_shift_mean': _mean(self._public_target_shifts), 'public_target_shift_max': None if not self._public_target_shifts else float(max(self._public_target_shifts)), 'environment_public_target_at_handoff': self.environment_public_target_at_handoff, 'controller_execution_target_at_handoff': self.controller_execution_target_at_handoff, 'installed_executor_tracking_target_at_handoff': self.installed_executor_tracking_target_at_handoff, 'controller_to_public_target_error_at_handoff': self.controller_to_public_target_error_at_handoff, 'controller_to_public_target_error_mean': _mean(self._controller_to_public_target_errors), 'controller_to_public_target_error_final': None if not self._controller_to_public_target_errors else float(self._controller_to_public_target_errors[-1]), 'full_planning_refresh_count': int(self.full_planning_refresh_count), 'handoff_forced_refresh_count': int(self.handoff_forced_refresh_count), 'target_shift_forced_refresh_count': int(self.target_shift_forced_refresh_count), 'post_found_collision_count': int(self.post_found_collision_count), 'pre_found_base_reward': float(self.pre_found_base_reward), 'post_found_base_reward': float(self.post_found_base_reward), 'post_found_shaping_reward': float(self.post_found_shaping_reward), 'completion_reward_post_tanh': float(self.completion_reward_post_tanh), 'adjusted_episode_reward': float(self.adjusted_episode_reward), 'search_residual_ratio_pre_found': _mean(self._search_residual_pre), 'search_residual_ratio_post_found': _mean(self._search_residual_post), 'executor_residual_ratio_pre_found': _mean(self._executor_residual_pre), 'executor_residual_ratio_post_found': _mean(self._executor_residual_post), 'initial_planner_endpoint_fallback_count': initial_fallback}
