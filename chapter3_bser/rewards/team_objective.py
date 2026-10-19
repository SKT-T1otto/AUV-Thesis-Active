from __future__ import annotations
import numpy as np
import torch
INDIVIDUAL = 'individual_v1'
TEAM = 'team_mean_v1'
SOURCE_REVISION = 'execution_reward_v2_after_terminal_override_v1'
OBJECTIVE_FIELDS = ('reward_objective', 'source_reward_revision')

def objective_identity(config=None):
    config = {} if config is None else config
    objective = config.get('reward_objective', INDIVIDUAL)
    if objective not in (INDIVIDUAL, TEAM):
        raise ValueError(f'unknown reward_objective: {objective}')
    revision = config.get('source_reward_revision', SOURCE_REVISION)
    if revision != SOURCE_REVISION:
        raise ValueError(f'unknown source_reward_revision: {revision}')
    return dict(reward_objective=objective, source_reward_revision=revision)

class RewardAccounting:

    def __init__(self, config=None, gamma=0.95):
        self.identity = objective_identity(config)
        self.gamma = float(gamma)
        if not 0 < self.gamma <= 1:
            raise ValueError('gamma must be in (0, 1]')
        self.reset()

    def reset(self):
        self.steps = 0
        self.team_undiscounted_return = 0.0
        self.team_discounted_return = 0.0
        self.source_individual_returns = np.zeros(4, dtype=np.float64)
        self.last = {}

    def apply(self, rewards, breakdown):
        if breakdown.get('reward_transform_applied_count', 0):
            raise RuntimeError('final reward transform already applied')
        source = torch.as_tensor(rewards).detach().clone().reshape(-1)
        if not source.is_floating_point():
            source = source.float()
        if source.shape != (4,) or not bool(torch.isfinite(source).all()):
            raise ValueError('source reward must be a finite four-vector')
        team = source.mean()
        shared = self.identity['reward_objective'] == TEAM
        final = team.repeat(4) if shared else source
        self.last = dict(**self.identity, source_reward_by_agent=source.cpu().tolist(), team_reward=float(team.item()), final_reward_by_agent=final.cpu().tolist(), reward_transform_applied_count=int(shared))
        self.source_individual_returns += source.cpu().numpy()
        self.team_undiscounted_return += float(team.item())
        self.team_discounted_return += self.gamma ** self.steps * float(team.item())
        self.steps += 1
        return final.cpu().numpy().copy() if isinstance(rewards, np.ndarray) else final

    def summary(self):
        return dict(**self.identity, gamma=self.gamma, team_undiscounted_return=self.team_undiscounted_return, team_discounted_return=self.team_discounted_return, source_individual_returns=self.source_individual_returns.tolist())
