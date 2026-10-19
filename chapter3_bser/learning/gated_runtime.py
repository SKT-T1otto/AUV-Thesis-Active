import numpy as np
import torch
from chapter3_bser.learning.base_runtime import BaselineMissionRuntime
from chapter3_bser.learning.config import VERSION

def execution_gates(observations, guidance, *, mask_hold):
    obs = np.asarray(observations)
    if obs.shape != (4, 28) or not np.isfinite(obs).all():
        raise ValueError('execution gate requires the original 4 x 28 observations')
    gates = np.ones(4, dtype=bool)
    if mask_hold:
        assignments = {int(item.agent_id): item for item in guidance.agent_assignments}
        if len(guidance.agent_assignments) != 4 or set(assignments) != set(range(4)):
            raise ValueError('all four guidance assignments are required')
        for (i, item) in assignments.items():
            gates[i] = bool(item.reachable and (not item.hold_state))
    return gates

class RepairedMissionRuntime(BaselineMissionRuntime):

    def _gates(self):
        gates = execution_gates(self.observations, self.guidance, mask_hold=self.repair_config['repairs']['hold_residual'])
        if self.repair_config['repairs']['inactive_actor']:
            gates &= np.asarray(self.observations)[:, 20] < 0.5
        return gates

    def _actions_for_step(self, actions):
        self.proposed_actions = actions.clone()
        self.applied_gates = self._gates()
        self.repair_counts['disabled_agent_steps'] += int((~self.applied_gates).sum())
        self.repair_counts['suppressed_nonzero_residuals'] += int(((actions.abs().sum(1) > 0).numpy() & ~self.applied_gates).sum())
        return actions.clamp(-1.0, 1.0) * torch.as_tensor(self.applied_gates)[:, None]

    def advance(self, model, *, explore=True):
        phase = 'post_found' if self.env.get_task_state().target_found else 'pre_found'
        row = super().advance(model, explore=explore)
        row.update(action_enabled=self.applied_gates.copy(), next_action_enabled=np.zeros(4, dtype=bool) if self.terminal else self._gates(), proposed_actions=self.proposed_actions.numpy().copy())
        for (name, value) in self.env.unwrapped.last_reward_components.items():
            values = torch.as_tensor(value).cpu().numpy().astype(float)
            self.raw_components[name] = (np.asarray(self.raw_components.get(name, np.zeros(4))) + values).tolist()
        self.phase_returns[phase] += row['team_reward']
        return row

    def summary(self):
        row = super().summary()
        row.update(repair_version=VERSION, repair_switches=self.repair_config['repairs'], waypoint_reward_timing=self.config['waypoint_reward_timing'], action_safety=dict(self.repair_counts), raw_pre_tanh_components=self.raw_components, team_return_by_phase=dict(self.phase_returns), observation_risks={'input_contract': [28, 3, 124], 'remaining_time_observed': False, 'previous_acceleration_observed': False, 'full_map_observed': False, 'execution_gate_is_network_input': False, 'markov_sufficiency_established': False}, safety_guarantee=False, reward_metric_equivalence_established=False)
        return row
