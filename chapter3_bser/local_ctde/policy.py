"""Deployment has one local actor. Only the trainer owns a joint value function."""
import numpy as np
import torch
from torch import nn

from core.algorithms.networks import MLPNetwork


class GoalPolicy(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.mean = MLPNetwork(28, 3, hidden_dim=width)
        # Fixed per-role exploration scales inherited from the native learner.
        self.register_buffer('role_scale', torch.tensor([.18, .14, .10, .08]))

    def distribution(self, features):
        scale = (features[..., 18:22] @ self.role_scale).unsqueeze(-1)
        return torch.distributions.Normal(self.mean(features), scale)

    @torch.no_grad()
    def propose(self, features, generator=None):
        row = torch.as_tensor(features, dtype=torch.float32)
        if row.shape != (28,):
            raise ValueError('each deployed actor requires a single local observation of size 28')
        dist = self.distribution(row)
        latent = dist.mean
        if generator is not None:
            latent = latent + dist.scale * torch.randn((3,), generator=generator)
        return latent.numpy().copy(), torch.tanh(latent).numpy().copy()


def discounted_targets(rewards, gamma, final_value):
    value = float(final_value)
    targets = np.empty(len(rewards), dtype=np.float32)
    for t in reversed(range(len(rewards))):
        value = float(rewards[t]) + gamma * value
        targets[t] = value
    return targets


class TeamTrainer:
    """One on-policy update per rollout, score only actual high-level draws."""

    def __init__(self, rl):
        self.actor = GoalPolicy(rl['hidden_dim'])
        self.critic = MLPNetwork(124, 1, hidden_dim=rl['hidden_dim'])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=rl['lr_actor'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=rl['lr_critic'])
        self.gamma = rl['gamma']

    def update(self, episode):
        states = torch.tensor(np.stack(episode['states']), dtype=torch.float32)
        values = self.critic(states).flatten()
        with torch.no_grad():
            tail = 0. if episode['terminated'] else float(self.critic(
                torch.tensor(episode['final_state'], dtype=torch.float32)).item())
            target = torch.from_numpy(discounted_targets(episode['rewards'], self.gamma, tail))
            advantage = target - values
        decisions = episode['decisions']
        if not decisions:
            raise ValueError('no on-policy search-goal decisions to update')
        times = torch.tensor([d['step'] for d in decisions], dtype=torch.long)
        features = torch.tensor([d['features'] for d in decisions], dtype=torch.float32)
        latents = torch.tensor([d['latent'] for d in decisions], dtype=torch.float32)
        logp = self.actor.distribution(features).log_prob(latents).sum(dim=-1)
        # Each trajectory contributes a sum over the actual decisions, including
        # rejected proposals. Gradients use the sampled latent before projection.
        actor_loss = -(self.gamma ** times * advantage[times] * logp).sum()
        critic_loss = (values - target).square().mean()
        if not torch.isfinite(torch.stack((actor_loss, critic_loss))).all():
            raise ValueError('nonfinite on-policy update')
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        return dict(actor_loss=actor_loss.item(), critic_loss=critic_loss.item(),
                    draws=len(times), tail_value=tail, critic_input_dim=states.shape[1])
