from __future__ import annotations
from typing import Iterable
from typing import Mapping
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.events.trigger_policy import TriggerDecision
from chapter3_bser.hysteresis.cooldown import ReplanningCooldown

class ReplanningPolicy:
    CRITICAL_EVENTS = frozenset((BSEREvent.TARGET_FOUND, BSEREvent.OBSTACLE_DISCOVERED))
    EVENT_PRIORITY = (BSEREvent.EXECUTOR_TARGET_RECEIVED, BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED, BSEREvent.EXECUTOR_INVALID, BSEREvent.OBSTACLE_DISCOVERED, BSEREvent.WAYPOINT_STALE, BSEREvent.BELIEF_SHIFT, BSEREvent.PERIODIC_REFRESH)

    def __init__(self, config: Mapping):
        specification = config['hysteresis']
        self.enabled = bool(specification.get('enabled', True))
        self.minimum_gain_threshold = float(specification['minimum_gain_threshold'])
        self.cooldown = ReplanningCooldown(int(specification['cooldown_steps']))
        self.mechanism_version = str(config.get('mechanism_version', 'phase1b_v1'))
        self.minimum_relative_gain = float(specification.get('minimum_relative_gain', self.minimum_gain_threshold))
        self.event_cooldowns = {BSEREvent.BELIEF_SHIFT: int(specification.get('belief_cooldown_steps', 20)), BSEREvent.OBSTACLE_DISCOVERED: int(specification.get('obstacle_cooldown_steps', 5)), BSEREvent.EXECUTOR_TARGET_RECEIVED: int(specification.get('target_received_cooldown_steps', 0)), BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED: int(specification.get('public_target_update_cooldown_steps', config.get('execution_runtime', {}).get('public_target_update_min_steps', 20))), BSEREvent.EXECUTOR_INVALID: int(specification.get('executor_invalid_cooldown_steps', 0)), BSEREvent.WAYPOINT_STALE: int(specification.get('waypoint_stale_cooldown_steps', 5)), BSEREvent.PERIODIC_REFRESH: int(specification.get('belief_cooldown_steps', 20))}
        self.last_event_step: dict[BSEREvent, int] = {}

    def primary_event(self, events: Iterable[BSEREvent]) -> BSEREvent | None:
        values = set(events)
        return next((event for event in self.EVENT_PRIORITY if event in values), None)

    def event_cooldown_remaining(self, event: BSEREvent | None, step: int) -> int:
        if event is None or event not in self.last_event_step:
            return 0
        return max(0, self.event_cooldowns.get(event, 0) - (int(step) - self.last_event_step[event]))

    def decide(self, events: Iterable[BSEREvent], new_objective: float, old_objective: float, step: int) -> TriggerDecision:
        event_tuple = tuple(events)
        gain = float(new_objective) - float(old_objective)
        if self.mechanism_version in {'phase1b1_corrected', 'phase1b2_corrected'}:
            event = self.primary_event(event_tuple)
            remaining = self.event_cooldown_remaining(event, step)
            if event is None:
                return TriggerDecision(False, 'NO_REPLAN_EVENT', gain, 0)
            if event == BSEREvent.EXECUTOR_TARGET_RECEIVED:
                return TriggerDecision(True, 'ACCEPT_PUBLIC_HANDOFF', gain, 0)
            if remaining:
                return TriggerDecision(False, 'REJECT_EVENT_COOLDOWN', gain, remaining)
            if event == BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED:
                return TriggerDecision(True, 'EXECUTOR_PUBLIC_TARGET_UPDATED', gain, 0)
            if event == BSEREvent.EXECUTOR_INVALID:
                return TriggerDecision(True, 'ACCEPT_EXECUTOR_INVALID', gain, 0)
            if event in {BSEREvent.OBSTACLE_DISCOVERED, BSEREvent.WAYPOINT_STALE}:
                return TriggerDecision(True, f'ACCEPT_{event.value}', gain, 0)
            denominator = max(abs(float(old_objective)), 1e-12)
            relative_gain = gain / denominator
            if relative_gain <= self.minimum_relative_gain:
                return TriggerDecision(False, 'REJECT_INSUFFICIENT_RELATIVE_GAIN', gain, 0)
            return TriggerDecision(True, 'ACCEPT_RELATIVE_GAIN', gain, 0)
        critical = next((event for event in event_tuple if event in self.CRITICAL_EVENTS), None)
        if critical is not None:
            return TriggerDecision(True, f'critical:{critical.value}', gain, self.cooldown.remaining(step))
        if not event_tuple:
            return TriggerDecision(False, 'no_event', gain, self.cooldown.remaining(step))
        if not self.enabled:
            return TriggerDecision(True, 'hysteresis_disabled', gain, 0)
        remaining = self.cooldown.remaining(step)
        if remaining:
            return TriggerDecision(False, 'cooldown_active', gain, remaining)
        if gain <= self.minimum_gain_threshold:
            return TriggerDecision(False, 'insufficient_gain', gain, 0)
        return TriggerDecision(True, 'gain_and_cooldown', gain, 0)

    def mark_replan(self, step: int, event: BSEREvent | None=None) -> None:
        self.cooldown.mark(step)
        if event is not None:
            self.last_event_step[event] = int(step)

    def mark_attempt(self, step: int, event: BSEREvent | None) -> None:
        if event is not None:
            self.last_event_step[event] = int(step)
