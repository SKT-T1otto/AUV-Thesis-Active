"""Keep the existing event schedule and safety recovery around map scoring."""
from chapter3_bser.d2.scored_controller import EffectController, EffectPolicy
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.events.trigger_policy import TriggerDecision


class CoupledPolicy(EffectPolicy):
    def __init__(self, config, allocator):
        super().__init__(config);self.allocator=allocator

    def decide(self, events, new_objective, old_objective, step):
        decision=super().decide(events,new_objective,old_objective,step)
        if (decision.should_replan and self.search_enabled and self.allocator.core_active
                and not self.recovery_ready
                and abs(new_objective-old_objective)<=self.allocator.research['tie_tolerance']):
            return TriggerDecision(False,'COUPLED_PRESERVE_REFERENCE_TIE',float(new_objective-old_objective),0)
        return decision


class CoupledController(EffectController):
    def step(self,state,mission_context=None):
        detection=self.detector.detect(self.cache.current,state,self.current_context,mission_context,
                                       assignment=self.current_allocation)
        primary=self.policy.primary_event(detection.events)
        eligible=primary is not None and not self.policy.event_cooldown_remaining(primary,state.step)
        if not state.target_found and (eligible or self.policy.recovery_ready):
            return super().step(state,mission_context)
        return OnlineBSERController.step(self,state,mission_context)
