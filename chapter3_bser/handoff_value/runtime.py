"""Read-only handoff instrumentation for the synchronized final-leg D2 runtime."""

from chapter3_bser.handoff_value.records import EpisodeRecords
from chapter3_bser.r123_final_leg import RuleRuntime as FinalLegRuntime
from chapter3_bser.r123_task_sync.receipts import snapshot


class RuleRuntime(FinalLegRuntime):
    def __init__(self, config, scenario, *, seed, episode_id=0, **kwargs):
        # The parent constructs an EffectAllocator and solves before its manager exists.
        self.handoff_records = EpisodeRecords(scenario["scenario_id"], episode_id)
        super().__init__(config, scenario, seed=seed, episode_id=episode_id, **kwargs)

    def d2_proposal_hook(self, state, candidates, frozen, standbys, result, trigger):
        initializing = getattr(self, "manager", None) is None
        receipts = snapshot(self, state) if frozen and not initializing else {}
        return self.handoff_records.record_proposal(
            self.step, state, candidates, frozen, standbys, result, trigger, receipts,
            initializing=initializing,
        )

    def advance(self, model, *, explore=False):
        transition = super().advance(model, explore=explore)
        self.handoff_records.end_step(self.step, transition["team_reward"], self.env)
        return transition

    def summary(self):
        result = super().summary()
        result["handoff_value"] = self.handoff_records.export(
            self.env.get_task_state(), self.env.get_episode_result(), self.found_step,
        )
        return result
