from __future__ import annotations
from typing import Any
from typing import Mapping
import torch
from chapter3_bser.protocols import ExecutionEpisodeDiagnostics
from chapter3_bser.protocols import Phase1CTransitionMetadata
from chapter3_bser.rewards.reward_adapter import Phase1CExecutionRewardAdapter
from chapter3_bser.rewards.reward_adapter import Phase1CExecutionRewardConfig

class Phase1CV2TrainingEnv:

    def __init__(self, env: Any, *, reward_config: Phase1CExecutionRewardConfig | Mapping[str, Any] | None=None) -> None:
        self.env = env
        self.reward_adapter = Phase1CExecutionRewardAdapter(reward_config)
        self.diagnostics = ExecutionEpisodeDiagnostics()
        self._episode_id = 0
        self._episode_index = 0
        self._scenario_id: str | None = None
        self._scenario_seed: int | None = None
        self._last_task = None
        self._previous_contact_total = 0
        self._previous_full_hold_total = 0
        self._previous_hold_counter = 0
        self.last_transition_metadata: Phase1CTransitionMetadata | None = None
        self.last_reward_breakdown: dict[str, Any] = {}
        self.last_base_rewards: torch.Tensor | None = None
        self._closed = False

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def installed_context(self):
        return getattr(self.env, 'installed_context', None)

    @property
    def installed_allocation_version(self):
        return getattr(self.env, 'installed_allocation_version', None)

    @staticmethod
    def _counter(runtime: Any, name: str, default: int=0) -> int:
        try:
            return int(getattr(runtime, name, default))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _hold_counter(runtime: Any) -> int:
        if hasattr(runtime, '_capture_hold_counter'):
            try:
                return int(getattr(runtime, '_capture_hold_counter'))
            except (TypeError, ValueError):
                pass
        counters = getattr(runtime, 'hold_counters', None)
        if counters is not None:
            try:
                return int(torch.as_tensor(counters).reshape(-1)[3].item())
            except (TypeError, ValueError, RuntimeError, IndexError):
                pass
        return 0

    def reset(self, scenario=None, *, episode_id: int=0, episode_index: int=0):
        self.reward_adapter.reset()
        self._episode_id = int(episode_id)
        self._episode_index = int(episode_index)
        scenario_dict = {} if scenario is None else dict(scenario)
        self._scenario_id = None if scenario is None else str(scenario_dict.get('scenario_id', ''))
        self._scenario_seed = None if scenario is None or scenario_dict.get('scenario_seed') is None else int(scenario_dict['scenario_seed'])
        observations = self.env.reset(scenario=scenario)
        runtime = self.unwrapped
        self._previous_contact_total = self._counter(runtime, 'capture_contact_step_count', 0)
        self._previous_full_hold_total = self._counter(runtime, 'capture_full_hold_step_count', 0)
        self._previous_hold_counter = self._hold_counter(runtime)
        self._last_task = self.get_task_state()
        self.last_transition_metadata = None
        self.last_reward_breakdown = {}
        self.last_base_rewards = None
        self.diagnostics.reset(self, episode_id=self._episode_id, episode_index=self._episode_index, scenario_id=self._scenario_id, scenario_seed=self._scenario_seed, max_steps=int(getattr(runtime, 'max_steps', 0)))
        return observations

    def step(self, actions):
        if self._last_task is None:
            raise RuntimeError('Phase1CV2TrainingEnv.reset must be called before step')
        task_before = self._last_task
        runtime = self.unwrapped
        hold_before = self._previous_hold_counter
        contact_total_before = self._previous_contact_total
        full_hold_total_before = self._previous_full_hold_total
        (observations, base_rewards, dones) = self.env.step(actions)
        task_after = self.get_task_state()
        contact_total_after = self._counter(runtime, 'capture_contact_step_count', 0)
        full_hold_total_after = self._counter(runtime, 'capture_full_hold_step_count', 0)
        hold_after = self._hold_counter(runtime)
        contact = contact_total_after > contact_total_before
        full_hold = full_hold_total_after > full_hold_total_before
        if not contact:
            swept = getattr(runtime, 'capture_swept_min_distance', None)
            radius = getattr(runtime, 'target_capture_radius', None)
            try:
                contact = swept is not None and radius is not None and (float(swept) <= float(radius)) and bool(getattr(task_after, 'target_found', False))
            except (TypeError, ValueError):
                contact = False
        result = self.reward_adapter.adjust(base_rewards, task_before=task_before, task_after=task_after, runtime=runtime, contact=contact, full_hold=full_hold, hold_counter_before=hold_before, hold_counter_after=hold_after, episode_id=self._episode_id, episode_index=self._episode_index)
        if result.metadata is None:
            raise RuntimeError('Phase 1C-v2 reward adapter did not emit metadata')
        self.last_transition_metadata = result.metadata
        self.last_reward_breakdown = dict(result.breakdown)
        self.last_base_rewards = torch.as_tensor(base_rewards, dtype=torch.float32).clone()
        self.diagnostics.observe_step(self, result.rewards, task_before=task_before, task_after=task_after, base_rewards=base_rewards, reward_breakdown=result.breakdown)
        self._last_task = task_after
        self._previous_contact_total = contact_total_after
        self._previous_full_hold_total = full_hold_total_after
        self._previous_hold_counter = hold_after
        if getattr(self.env, 'return_numpy', False):
            public_rewards = result.rewards.detach().cpu().numpy().copy()
        else:
            public_rewards = result.rewards.detach().clone()
        return (observations, public_rewards, dones)

    def observe_controller_result(self, result: Any, *, controller: Any | None=None, state_provider: Any | None=None) -> None:
        """Attach controller-only event diagnostics after a step.

        This is diagnostics-only and does not alter the transition metadata or
        reward already returned by ``step``.
        """
        if result is None:
            return
        self.diagnostics.observe_controller_result(self, result, controller=controller, state_provider=state_provider)

    def finalize_episode(self) -> dict[str, Any]:
        row = self.diagnostics.finalize(self)
        if getattr(self.unwrapped, 'task_protocol', 'legacy_nonterminal_v1') == 'collision_terminal_v1':
            row.update(self.unwrapped.get_episode_result())
        row.update({'contact_bonus_count': int(self.reward_adapter.contact_bonus_count), 'hold_bonus_count': int(self.reward_adapter.hold_bonus_count), 'terminal_bonus_count': int(self.reward_adapter.terminal_bonus_count), 'discovery_correction_count': int(self.reward_adapter.discovery_correction_count), 'searcher_zeroed_step_count': int(self.reward_adapter.searcher_zeroed_step_count)})
        return row

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.env, 'close', None)
        if callable(close):
            close()

    def __getattr__(self, name: str):
        return getattr(self.env, name)
