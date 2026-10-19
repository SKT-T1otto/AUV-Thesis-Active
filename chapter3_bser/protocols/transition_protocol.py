from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from typing import Mapping
from core.env.task_protocol import LEGACY
from core.env.task_protocol import protocol_identity
from chapter3_bser.rewards.team_objective import INDIVIDUAL
from chapter3_bser.rewards.team_objective import OBJECTIVE_FIELDS
from chapter3_bser.rewards.team_objective import objective_identity
from chapter3_bser.protocols import Phase1CTransitionMetadata
from chapter3_bser.protocols.stage_mapping import PRRACStage
from chapter3_bser.protocols.stage_mapping import transition_phase_to_prrac_stage

@dataclass(frozen=True)
class PRRACTransitionMetadata:
    base: Phase1CTransitionMetadata
    stage_before: PRRACStage
    stage_after: PRRACStage
    task_protocol: str = LEGACY
    termination_reason: str = 'running'
    terminated: bool = False
    truncated: bool = False
    reward_objective: str = INDIVIDUAL
    objective_metadata_present: bool = False
    source_reward_by_agent: tuple[float, ...] | None = None
    team_reward: float | None = None
    final_reward_by_agent: tuple[float, ...] | None = None
    reward_transform_applied_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base, Phase1CTransitionMetadata):
            raise TypeError('PRRAC metadata base must be Phase1CTransitionMetadata')
        object.__setattr__(self, 'stage_before', PRRACStage(int(self.stage_before)))
        object.__setattr__(self, 'stage_after', PRRACStage(int(self.stage_after)))
        expected_after = transition_phase_to_prrac_stage(self.base.phase)
        if self.stage_after != expected_after:
            raise ValueError(f'stage_after {self.stage_after.name} does not match base phase {self.base.phase.name}')

    def to_dict(self) -> dict[str, Any]:
        objective_fields = {}
        if self.objective_metadata_present or self.reward_objective != INDIVIDUAL:
            objective_fields = {**objective_identity({'reward_objective': self.reward_objective}), 'source_reward_by_agent': self.source_reward_by_agent, 'team_reward': self.team_reward, 'final_reward_by_agent': self.final_reward_by_agent, 'reward_transform_applied_count': self.reward_transform_applied_count}
        return {**protocol_identity({'task_protocol': self.task_protocol}), **objective_fields, 'termination_reason': self.termination_reason, 'terminated': self.terminated, 'truncated': self.truncated, 'base': self.base.to_dict(), 'stage_before': int(self.stage_before), 'stage_before_name': self.stage_before.name, 'stage_after': int(self.stage_after), 'stage_after_name': self.stage_after.name}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'PRRACTransitionMetadata':
        return cls(base=Phase1CTransitionMetadata.from_dict(value['base']), stage_before=PRRACStage(int(value['stage_before'])), stage_after=PRRACStage(int(value['stage_after'])), task_protocol=protocol_identity(value)['task_protocol'], termination_reason=str(value.get('termination_reason', 'running')), terminated=bool(value.get('terminated', False)), truncated=bool(value.get('truncated', False)), reward_objective=objective_identity(value)['reward_objective'], objective_metadata_present=any((key in value for key in OBJECTIVE_FIELDS)), source_reward_by_agent=None if value.get('source_reward_by_agent') is None else tuple(value['source_reward_by_agent']), team_reward=value.get('team_reward'), final_reward_by_agent=None if value.get('final_reward_by_agent') is None else tuple(value['final_reward_by_agent']), reward_transform_applied_count=int(value.get('reward_transform_applied_count', 0)))
