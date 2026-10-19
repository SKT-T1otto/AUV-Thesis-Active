from __future__ import annotations
from typing import Any
from chapter3_bser.rewards.team_objective import RewardAccounting
from chapter3_bser.rewards.team_objective import TEAM
from chapter3_bser.rewards.team_objective import OBJECTIVE_FIELDS
from chapter3_bser.protocols import TransitionPhase
from chapter3_bser.protocols.transition_protocol import PRRACTransitionMetadata
from chapter3_bser.protocols.stage_mapping import transition_phase_to_prrac_stage

class PRRACTrainingEnv:

    def __init__(self, env: Any, *, reward_objective_config=None, gamma=0.95) -> None:
        self.env = env
        self.objective_explicit = any((key in (reward_objective_config or {}) for key in OBJECTIVE_FIELDS))
        self.reward_accounting = RewardAccounting(reward_objective_config, gamma)
        self._previous_phase = TransitionPhase.PRE_FOUND
        self.last_prrac_transition_metadata: PRRACTransitionMetadata | None = None
        self._closed = False

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def installed_context(self):
        return getattr(self.env, 'installed_context', None)

    @property
    def last_reward_breakdown(self):
        return getattr(self.env, 'last_reward_breakdown', {})

    @property
    def last_transition_metadata(self):
        return getattr(self.env, 'last_transition_metadata', None)

    def reset(self, *args, **kwargs):
        self.reward_accounting.reset()
        self._previous_phase = TransitionPhase.PRE_FOUND
        self.last_prrac_transition_metadata = None
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        stage_before = transition_phase_to_prrac_stage(self._previous_phase)
        (observations, source_rewards, dones) = self.env.step(actions)
        rewards = self.reward_accounting.apply(source_rewards, self.last_reward_breakdown)
        if self.objective_explicit:
            self.last_reward_breakdown.update(self.reward_accounting.last)
        base = self.last_transition_metadata
        if base is None:
            raise RuntimeError('Phase1CV2TrainingEnv did not emit transition metadata')
        current_phase = base.phase
        stage_after = transition_phase_to_prrac_stage(current_phase)
        self.last_prrac_transition_metadata = PRRACTransitionMetadata(base=base, stage_before=stage_before, stage_after=stage_after, task_protocol=getattr(self.unwrapped, 'task_protocol', 'legacy_nonterminal_v1'), reward_objective=self.reward_accounting.identity['reward_objective'], objective_metadata_present=self.objective_explicit, source_reward_by_agent=tuple(self.reward_accounting.last['source_reward_by_agent']), team_reward=self.reward_accounting.last['team_reward'], final_reward_by_agent=tuple(self.reward_accounting.last['final_reward_by_agent']), reward_transform_applied_count=self.reward_accounting.last['reward_transform_applied_count'], termination_reason=getattr(getattr(self.unwrapped, 'episode_outcome', None), 'termination_reason', 'running'), terminated=getattr(getattr(self.unwrapped, 'episode_outcome', None), 'episode_terminated', False))
        self._previous_phase = current_phase
        return (observations, rewards, dones)

    def finalize_episode(self):
        row = self.env.finalize_episode()
        if self.objective_explicit:
            row.update(self.reward_accounting.summary())
            row['source_searcher_zeroed_step_count'] = row.get('searcher_zeroed_step_count', 0)
        if self.reward_accounting.identity['reward_objective'] == TEAM:
            row['source_adjusted_episode_reward'] = row.get('adjusted_episode_reward')
            row['adjusted_episode_reward'] = self.reward_accounting.team_undiscounted_return
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
