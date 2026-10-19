from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Any
from typing import Mapping

class TransitionPhase(IntEnum):
    """Mutually exclusive Phase 1C transition classes."""
    PRE_FOUND = 0
    POST_FOUND = 1
    CONTACT = 2
    HOLD = 3
    SUCCESS = 4

def classify_transition_phase(*, task_found: bool, contact: bool, full_hold: bool, mission_complete: bool) -> TransitionPhase:
    """Classify a transition using the frozen priority order.

    Priority: SUCCESS > HOLD > CONTACT > POST_FOUND > PRE_FOUND.
    """
    if bool(mission_complete):
        return TransitionPhase.SUCCESS
    if bool(full_hold):
        return TransitionPhase.HOLD
    if bool(contact):
        return TransitionPhase.CONTACT
    if bool(task_found):
        return TransitionPhase.POST_FOUND
    return TransitionPhase.PRE_FOUND

@dataclass(frozen=True)
class Phase1CTransitionMetadata:
    episode_id: int
    episode_index: int
    step: int
    task_found: bool
    executor_target_assigned: bool
    contact: bool
    full_hold: bool
    hold_counter: int
    mission_complete: bool
    phase: TransitionPhase

    @classmethod
    def build(cls, *, episode_id: int, episode_index: int, step: int, task_found: bool, executor_target_assigned: bool, contact: bool, full_hold: bool, hold_counter: int, mission_complete: bool) -> 'Phase1CTransitionMetadata':
        return cls(episode_id=int(episode_id), episode_index=int(episode_index), step=int(step), task_found=bool(task_found), executor_target_assigned=bool(executor_target_assigned), contact=bool(contact), full_hold=bool(full_hold), hold_counter=max(0, int(hold_counter)), mission_complete=bool(mission_complete), phase=classify_transition_phase(task_found=task_found, contact=contact, full_hold=full_hold, mission_complete=mission_complete))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation (compatible with allow_nan=False)."""
        return {'episode_id': int(self.episode_id), 'episode_index': int(self.episode_index), 'step': int(self.step), 'task_found': bool(self.task_found), 'executor_target_assigned': bool(self.executor_target_assigned), 'contact': bool(self.contact), 'full_hold': bool(self.full_hold), 'hold_counter': int(self.hold_counter), 'mission_complete': bool(self.mission_complete), 'phase': int(self.phase), 'phase_name': self.phase.name}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'Phase1CTransitionMetadata':
        phase_value = value.get('phase', value.get('phase_name'))
        if isinstance(phase_value, str):
            phase = TransitionPhase[phase_value]
        else:
            phase = TransitionPhase(int(phase_value))
        result = cls(episode_id=int(value['episode_id']), episode_index=int(value['episode_index']), step=int(value['step']), task_found=bool(value['task_found']), executor_target_assigned=bool(value['executor_target_assigned']), contact=bool(value['contact']), full_hold=bool(value['full_hold']), hold_counter=max(0, int(value['hold_counter'])), mission_complete=bool(value['mission_complete']), phase=phase)
        expected = classify_transition_phase(task_found=result.task_found, contact=result.contact, full_hold=result.full_hold, mission_complete=result.mission_complete)
        if result.phase != expected:
            raise ValueError(f'transition phase mismatch: stored={result.phase.name}, classified={expected.name}')
        return result
