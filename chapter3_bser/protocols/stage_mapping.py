from __future__ import annotations
from enum import IntEnum
from chapter3_bser.protocols.transition_schema import TransitionPhase

class PRRACStage(IntEnum):
    SEARCH = 0
    INTERCEPT = 1
    HOLD = 2
_TRANSITION_TO_PRRAC = {TransitionPhase.PRE_FOUND: PRRACStage.SEARCH, TransitionPhase.POST_FOUND: PRRACStage.INTERCEPT, TransitionPhase.CONTACT: PRRACStage.HOLD, TransitionPhase.HOLD: PRRACStage.HOLD, TransitionPhase.SUCCESS: PRRACStage.HOLD}

def transition_phase_to_prrac_stage(phase: TransitionPhase | int | str) -> PRRACStage:
    """Map a stored transition phase without consulting future environment state."""
    if isinstance(phase, str):
        try:
            phase = TransitionPhase[phase]
        except KeyError as exc:
            raise ValueError(f'unknown transition phase: {phase!r}') from exc
    try:
        normalized = TransitionPhase(int(phase))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'unknown transition phase: {phase!r}') from exc
    return _TRANSITION_TO_PRRAC[normalized]
