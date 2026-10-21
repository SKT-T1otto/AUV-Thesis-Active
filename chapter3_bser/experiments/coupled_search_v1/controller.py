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
        # Refresh the installed routes' utility from the current joint belief
        # and observed execution errors at every search step. Replanning still
        # follows the existing events/cooldowns and safety stabilization.
        if not state.target_found:
            result=super().step(state,mission_context)
            if int(state.step) not in self.allocator.scored_steps:
                raise RuntimeError('coupled_search_scoring_unavailable: step='+str(state.step))
            return result
        return OnlineBSERController.step(self,state,mission_context)
