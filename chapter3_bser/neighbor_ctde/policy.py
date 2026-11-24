"""The local 28-D backbone plus a separate, on-board message input."""
import numpy as np
import torch
from torch import nn
from chapter3_bser.local_ctde.policy import TeamTrainer
from .observations import CONTEXT_DIM


class NeighborPolicy(nn.Module):
    def __init__(self, local_policy):
        super().__init__()
        self.local = local_policy
        self.from_messages = nn.Linear(CONTEXT_DIM, 28, bias=False)

    def distribution(self, features):
        if features.shape[-1] != 28 + CONTEXT_DIM:
            raise ValueError('expected one local input and three received-peer slots plus coverage')
        own, received = features.split([28, CONTEXT_DIM], dim=-1)
        mean = self.local.mean(own + self.from_messages(received))
        scale = (own[..., 18:22] @ self.local.role_scale).unsqueeze(-1)
        return torch.distributions.Normal(mean, scale)

    @torch.no_grad()
    def propose(self, features, generator=None):
        x = torch.tensor(features, dtype=torch.float32)
        if x.shape != (28 + CONTEXT_DIM,):
            raise ValueError('deployment accepts only a single addressed boat observation')
        dist = self.distribution(x)
        z = dist.mean
        if generator is not None:
            z = z + dist.scale * torch.randn(3, generator=generator)
        return z.numpy().copy(), z.tanh().numpy().copy()


def make_trainer(rl):
    trainer = TeamTrainer(rl)
    trainer.actor = NeighborPolicy(trainer.actor)
    trainer.actor_opt = torch.optim.Adam(trainer.actor.parameters(), lr=rl['lr_actor'])
    return trainer


class ReceivedPolicy:
    """Bind only this boat's inbox to the existing navigator's goal interface."""

    def __init__(self, actor, context):
        self.actor, self.context = actor, context

    def propose(self, own, generator):
        return self.actor.propose(np.concatenate((own, self.context)), generator)
