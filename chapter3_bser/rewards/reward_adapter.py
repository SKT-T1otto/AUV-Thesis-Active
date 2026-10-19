from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from typing import Mapping
import torch
from core.env.task_protocol import LEGACY
from core.env.task_protocol import STRICT
from core.env.task_protocol import validate_task_config
from chapter3_bser.protocols import Phase1CTransitionMetadata

@dataclass(frozen=True)
class Phase1CExecutionRewardConfig:
    enabled: bool = True
    freeze_searchers_after_found: bool = True
    preserve_discovery_reward: bool = True
    contact_entry_bonus: float = 0.25
    hold_increment_bonus: float = 0.2
    terminal_success_bonus: float = 2.0
    reward_clip: float = 3.0
    executor_id: int = 3
    searcher_ids: tuple[int, ...] = (0, 1, 2)
    schema: str = 'bser.phase1c.execution_reward.v2'

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> 'Phase1CExecutionRewardConfig':
        if value is None:
            return cls()
        mapping = dict(value)
        return cls(enabled=bool(mapping.get('enabled', True)), freeze_searchers_after_found=bool(mapping.get('freeze_searchers_after_found', mapping.get('freeze_searcher_rewards_after_found', True))), preserve_discovery_reward=bool(mapping.get('preserve_discovery_reward', mapping.get('preserve_discovery_event_reward', True))), contact_entry_bonus=float(mapping.get('contact_entry_bonus', 0.25)), hold_increment_bonus=float(mapping.get('hold_increment_bonus', 0.2)), terminal_success_bonus=float(mapping.get('terminal_success_bonus', mapping.get('terminal_success_bonus_post_tanh', 2.0))), reward_clip=float(mapping.get('reward_clip', mapping.get('reward_clip_abs', 3.0))), executor_id=int(mapping.get('executor_id', 3)), searcher_ids=tuple((int(item) for item in mapping.get('searcher_ids', (0, 1, 2)))), schema=str(mapping.get('schema', 'bser.phase1c.execution_reward.v2')))

    def to_dict(self) -> dict[str, Any]:
        return {'schema': self.schema, 'enabled': self.enabled, 'freeze_searchers_after_found': self.freeze_searchers_after_found, 'preserve_discovery_reward': self.preserve_discovery_reward, 'contact_entry_bonus': self.contact_entry_bonus, 'hold_increment_bonus': self.hold_increment_bonus, 'terminal_success_bonus': self.terminal_success_bonus, 'reward_clip': self.reward_clip, 'executor_id': self.executor_id, 'searcher_ids': list(self.searcher_ids)}

@dataclass(frozen=True)
class RewardAdjustmentResult:
    rewards: torch.Tensor
    breakdown: dict[str, Any]
    metadata: Phase1CTransitionMetadata | None = None

class Phase1CExecutionRewardAdapter:
    """Apply explicit v2 search-freeze and executor shaping semantics.

    No ground-truth target distance, prediction error, or oracle signal is used
    here.  Those values remain diagnostics-only.
    """

    def __init__(self, config: Phase1CExecutionRewardConfig | Mapping[str, Any] | None=None) -> None:
        self.config = config if isinstance(config, Phase1CExecutionRewardConfig) else Phase1CExecutionRewardConfig.from_mapping(config)
        self.reset()

    def reset(self) -> None:
        self._previous_contact = False
        self._terminal_bonus_emitted = False
        self.contact_bonus_count = 0
        self.hold_bonus_count = 0
        self.terminal_bonus_count = 0
        self.discovery_correction_count = 0
        self.searcher_zeroed_step_count = 0

    @staticmethod
    def _component(runtime: Any, name: str, *, length: int, device: torch.device) -> torch.Tensor | None:
        components = getattr(runtime, 'last_reward_components', None)
        if not isinstance(components, Mapping) or name not in components:
            return None
        value = torch.as_tensor(components[name], dtype=torch.float32, device=device).reshape(-1)
        if value.numel() != length:
            return None
        return value

    def _discovery_only_rewards(self, base: torch.Tensor, runtime: Any) -> tuple[torch.Tensor, bool]:
        find = self._component(runtime, 'reward_find_event', length=base.numel(), device=base.device)
        early = self._component(runtime, 'reward_early_find', length=base.numel(), device=base.device)
        if find is None or early is None:
            return (base.clone(), False)
        scale = float(getattr(runtime, 'reward_scale', 1.0))
        if not torch.isfinite(torch.tensor(scale)) or abs(scale) < 1e-12:
            scale = 1.0
        return (torch.tanh((find + early) / scale), True)

    def adjust(self, base_rewards: Any, *, task_before: Any, task_after: Any, runtime: Any, contact: bool, full_hold: bool, hold_counter_before: int, hold_counter_after: int, episode_id: int | None=None, episode_index: int | None=None) -> RewardAdjustmentResult:
        base = torch.as_tensor(base_rewards, dtype=torch.float32).reshape(-1).clone()
        if base.numel() < 4:
            raise ValueError(f'Phase 1C-v2 expects four rewards, got {base.numel()}')
        adjusted = base.clone()
        collision_terminal = getattr(runtime, 'task_protocol', LEGACY) == STRICT and runtime.episode_outcome.termination_reason == 'obstacle_collision'
        if collision_terminal:
            validate_task_config({**vars(runtime), 'reward': self.config.to_dict()})
            adjusted.fill_(runtime.collision_terminal_reward)
            metadata = Phase1CTransitionMetadata.build(episode_id=int(episode_id or 0), episode_index=int(episode_index or 0), step=int(getattr(task_after, 'step', 0)), task_found=bool(getattr(task_after, 'target_found', False)), executor_target_assigned=bool(getattr(task_after, 'executor_knows_target', False)), contact=False, full_hold=False, hold_counter=hold_counter_before, mission_complete=False)
            before = getattr(runtime, 'reward_before_terminal_override', base)
            return RewardAdjustmentResult(adjusted, {'base_reward': float(base.sum()), 'base_reward_by_agent': base.tolist(), 'raw_reward_components': {k: torch.as_tensor(v).tolist() for (k, v) in getattr(runtime, 'last_reward_components', {}).items()}, 'terminal_override': True, 'reward_before_terminal_override_by_agent': torch.as_tensor(before).tolist(), 'reward_after_terminal_override_by_agent': adjusted.tolist(), 'final_reward_by_agent': adjusted.tolist(), 'adjusted_reward': float(adjusted.sum()), 'adjusted_reward_by_agent': adjusted.tolist(), 'contact_entry_bonus': 0.0, 'hold_increment_bonus': 0.0, 'terminal_success_bonus': 0.0, 'terminal_success_bonus_post_tanh': 0.0, 'execution_shaping_total': 0.0}, metadata)
        before_found = bool(getattr(task_before, 'target_found', False))
        after_found = bool(getattr(task_after, 'target_found', False))
        before_complete = bool(getattr(task_before, 'mission_complete', False))
        after_complete = bool(getattr(task_after, 'mission_complete', False))
        discovery_transition = after_found and (not before_found)
        discovery_restored = 0.0
        discovery_components_available = False
        zeroed = False
        if self.config.enabled and self.config.freeze_searchers_after_found:
            searchers = list(self.config.searcher_ids)
            if discovery_transition and self.config.preserve_discovery_reward:
                (discovery, discovery_components_available) = self._discovery_only_rewards(base, runtime)
                adjusted[searchers] = discovery[searchers]
                discovery_restored = float(adjusted[searchers].sum().item())
                self.discovery_correction_count += 1
            elif not after_found:
                pass
            else:
                adjusted[searchers] = 0.0
                zeroed = True
                self.searcher_zeroed_step_count += 1
        contact_entry = bool(contact) and (not self._previous_contact)
        contact_bonus = self.config.contact_entry_bonus if self.config.enabled and contact_entry and after_found else 0.0
        hold_increment = max(0, int(hold_counter_after) - int(hold_counter_before))
        hold_bonus = self.config.hold_increment_bonus * hold_increment if self.config.enabled and after_found and bool(full_hold) and (hold_increment > 0) else 0.0
        terminal_transition = after_complete and (not before_complete)
        terminal_bonus = 0.0
        if self.config.enabled and terminal_transition and (not self._terminal_bonus_emitted):
            terminal_bonus = self.config.terminal_success_bonus
            self._terminal_bonus_emitted = True
            self.terminal_bonus_count += 1
        if contact_bonus:
            self.contact_bonus_count += 1
        if hold_bonus:
            self.hold_bonus_count += hold_increment
        executor = int(self.config.executor_id)
        adjusted[executor] = adjusted[executor] + contact_bonus + hold_bonus + terminal_bonus
        if self.config.enabled and self.config.reward_clip > 0.0:
            adjusted = torch.clamp(adjusted, -self.config.reward_clip, self.config.reward_clip)
        self._previous_contact = bool(contact)
        metadata = None
        if episode_id is not None and episode_index is not None:
            metadata = Phase1CTransitionMetadata.build(episode_id=int(episode_id), episode_index=int(episode_index), step=int(getattr(task_after, 'step', 0)), task_found=after_found, executor_target_assigned=bool(getattr(task_after, 'executor_knows_target', getattr(runtime, 'executor_target_assigned', False))), contact=bool(contact), full_hold=bool(full_hold), hold_counter=int(hold_counter_after), mission_complete=after_complete)
        breakdown = {'terminal_override': False, 'reward_before_terminal_override_by_agent': adjusted.tolist(), 'reward_after_terminal_override_by_agent': adjusted.tolist(), 'final_reward_by_agent': adjusted.tolist(), 'base_reward': float(base.sum().item()), 'base_reward_by_agent': [float(item) for item in base.tolist()], 'searcher_discovery_only_reward': float(discovery_restored), 'discovery_reward_restored': float(discovery_restored), 'discovery_components_available': bool(discovery_components_available), 'searcher_post_found_zeroed': bool(zeroed), 'contact_entry_bonus': float(contact_bonus), 'hold_increment': int(hold_increment), 'hold_increment_bonus': float(hold_bonus), 'terminal_success_bonus_post_tanh': float(terminal_bonus), 'terminal_success_bonus': float(terminal_bonus), 'execution_shaping_total': float(contact_bonus + hold_bonus + terminal_bonus), 'adjusted_reward': float(adjusted.sum().item()), 'adjusted_reward_by_agent': [float(item) for item in adjusted.tolist()]}
        return RewardAdjustmentResult(adjusted, breakdown, metadata)
