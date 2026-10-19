import torch

class ExecutionRepair:

    def _augment_sample(self, batch, sample):
        if not hasattr(sample, 'enabled') or not hasattr(sample, 'next_enabled'):
            raise ValueError('repair learner requires replay execution gates')
        for (name, raw, obs) in (('enabled', sample.enabled, batch['obs']), ('next_enabled', sample.next_enabled, batch['next_obs'])):
            value = torch.as_tensor(raw, device=self.device)
            if value.shape != (obs[0].shape[0], 4) or not bool(((value == 0) | (value == 1)).all()):
                raise ValueError('invalid replay execution gates')
            batch[name] = value.to(torch.float32)
        for (i, action) in enumerate(batch['acs']):
            if bool((action[batch['enabled'][:, i] == 0] != 0).any()):
                raise ValueError('critic replay contains an unapplied residual')

    def _project_policy_actions(self, batch, actions, *, next_state=False):
        gates = batch['next_enabled' if next_state else 'enabled']
        return [action * gates[:, i:i + 1] for (i, action) in enumerate(actions)]

    def _actor_sample_weights(self, batch, agent_i):
        weights = batch['enabled'][:, agent_i:agent_i + 1]
        if self.repairs['inactive_actor']:
            weights = weights * (batch['obs'][agent_i][:, 20:21] < 0.5)
        return weights

    def update(self, sample, agent_i, **kwargs):
        result = super().update(sample, agent_i, **kwargs)
        self.actor_updates[agent_i] += int(self.last_actor_updated)
        return result

    def _compute_target_q(self, *args):
        target = super()._compute_target_q(*args)
        self.last_target_diagnostics = dict(mean=float(target.mean()), abs_max=float(target.abs().max()), clipped_fraction=float((target.abs() == 10).float().mean()))
        return target
