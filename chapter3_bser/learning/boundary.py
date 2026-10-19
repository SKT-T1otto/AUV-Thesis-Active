from __future__ import annotations
from types import SimpleNamespace
import torch
from torch import nn
from core.algorithms.maddpg import MADDPG
from core.env.observation_contract import ROLE_ORDER
BOUNDARY_SCHEMA = 'ch3.baseline.boundary_encoder.v1'
BOUNDARY_FEATURE_DIM = 18

def boundary_features(observations: torch.Tensor) -> torch.Tensor:
    """Extract boundary state from the existing joint observation, never truth.

    Input/output shapes are [4, 28]/[4, 18] or [batch, 4, 28]/[batch, 4, 18].
    Detection is the persistent event latch conveyed by searcher knowledge;
    handoff is the executor's own knowledge flag. Target coordinates are gated
    separately for each actor, so discovery never supplies an unknown target
    location to the executor. Relative state is executor minus each searcher;
    for the executor it is searcher centroid minus executor. These public agent
    positions/velocities are already in the raw joint observations.
    """
    observations = torch.as_tensor(observations)
    if observations.ndim not in (2, 3) or tuple(observations.shape[-2:]) != (4, 28):
        raise ValueError('boundary conditioning requires joint 4 x 28 observations')
    single = observations.ndim == 2
    values = observations.unsqueeze(0) if single else observations
    known = values[..., 27:28]
    detection = known[:, :3].amax(dim=1, keepdim=True).expand(-1, 4, -1)
    handoff = known[:, 3:4].expand(-1, 4, -1)
    position = values[..., :3]
    velocity = values[..., 3:6]
    relative_position = torch.cat((position[:, 3:4] - position[:, :3], position[:, :3].mean(dim=1, keepdim=True) - position[:, 3:4]), dim=1)
    relative_velocity = torch.cat((velocity[:, 3:4] - velocity[:, :3], velocity[:, :3].mean(dim=1, keepdim=True) - velocity[:, 3:4]), dim=1)
    features = torch.cat((detection, handoff, relative_position, relative_velocity, values[..., 12:15] * known, values[..., 22:26], known, values[..., 20:21], values[..., 21:22]), dim=-1)
    return features.squeeze(0) if single else features

class BoundaryEncoder(nn.Module):
    """Trainable boundary embedding into the unchanged actor's 28D input."""
    schema = BOUNDARY_SCHEMA
    feature_dim = BOUNDARY_FEATURE_DIM
    output_dim = 28

    def __init__(self, hidden_dim: int=32):
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError('boundary encoder hidden_dim must be a positive integer')
        self.network = nn.Sequential(nn.LayerNorm(BOUNDARY_FEATURE_DIM), nn.Linear(BOUNDARY_FEATURE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 28), nn.Tanh())

    def forward(self, observations, boundary):
        if observations.shape[-1] != 28 or boundary.shape[-1] != BOUNDARY_FEATURE_DIM:
            raise ValueError('invalid boundary encoder input dimensions')
        return observations + self.network(boundary)

class BoundaryConditionedPolicy(nn.Module):
    """Wrap one unmodified core MLP with a boundary input encoder."""

    def __init__(self, actor: nn.Module, hidden_dim: int=32):
        super().__init__()
        self.actor = actor
        self.boundary_encoder = BoundaryEncoder(hidden_dim)
        self._boundary_context = None

    def bind_boundary(self, boundary: torch.Tensor):
        self._boundary_context = boundary

    def forward(self, observations):
        boundary = self._boundary_context
        if boundary is None or boundary.shape[:-1] != observations.shape[:-1]:
            raise RuntimeError('B3 policies require boundary context from the joint observations')
        return self.actor(self.boundary_encoder(observations, boundary))

class BoundaryConditionedMADDPG(MADDPG):
    """Only bind actor context; all learning remains in core MADDPG.

    Context is reconstructed from each replay sample, including next-state
    context for target policies. It is neither a rollout-only cached condition
    nor extra critic/replay dimensions. Checkpoints use build_model followed by
    load_training_state_dict, so wrapped actor and optimizer states are restored
    together and never loaded as an ordinary unconditioned core actor.
    """

    @staticmethod
    def _bind(policies, observations):
        features = boundary_features(torch.stack(observations, dim=1))
        for (index, policy) in enumerate(policies):
            policy.bind_boundary(features[:, index])

    def step(self, observations, explore=False):
        proc_obs = [self._ensure_obs_tensor(obs) for obs in observations]
        self._bind(self.policies, proc_obs)
        return super().step(proc_obs, explore=explore)

    def _build_target_actions(self, next_obs):
        self._bind(self.target_policies, next_obs)
        return super()._build_target_actions(next_obs)

    def _prepare_sample(self, sample):
        batch = super()._prepare_sample(sample)
        self._bind(self.policies, batch['obs'])
        return batch

def _contract_environment():
    """Fixed metadata permits checkpoint loading without resetting a simulator."""
    return SimpleNamespace(num_agents=4, observation_space={f'agent_{i}': SimpleNamespace(shape=(28,)) for i in range(4)}, action_space={f'agent_{i}': SimpleNamespace(shape=(3,)) for i in range(4)}, role_names=list(ROLE_ORDER), agent_specs=[{'name': role} for role in ROLE_ORDER])
