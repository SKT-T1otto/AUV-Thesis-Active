import torch
from core.replay.ch3_buffer import CH3ReplayBuffer

class ExecutionSample(tuple):

    def __new__(cls, values, enabled, next_enabled):
        sample = super().__new__(cls, values)
        sample.enabled = enabled
        sample.next_enabled = next_enabled
        return sample

class ExecutionReplay(CH3ReplayBuffer):

    def __init__(self, capacity):
        super().__init__(capacity, 4, [28] * 4, [3] * 4, success_priority=1.0, alpha=0.0, beta_start=0.0)
        self.enabled = torch.empty((capacity, 4), dtype=torch.bool)
        self.next_enabled = torch.empty_like(self.enabled)

    def push_transition(self, transition):
        gates = []
        for key in ('action_enabled', 'next_action_enabled'):
            value = torch.as_tensor(transition[key])
            if value.shape != (4,) or not bool(((value == 0) | (value == 1)).all()):
                raise ValueError('replay requires four binary execution gates')
            gates.append(value.bool())
        actions = torch.as_tensor(transition['actions'])
        if actions.shape != (4, 3) or not bool(torch.isfinite(actions).all()) or bool((actions.abs() > 1).any()) or bool((actions[~gates[0]] != 0).any()):
            raise ValueError('replay actions must be finite applied residuals, zero when disabled')
        index = self.next_idx
        super().push(transition['observations'], transition['actions'], transition['rewards'], transition['next_observations'], transition['dones'], transition.get('success_flags', [False] * 4))
        self.enabled[index].copy_(gates[0])
        self.next_enabled[index].copy_(gates[1])

    def sample(self, n, *, norm_rews=False, device='cpu'):
        if norm_rews:
            raise ValueError('repair replay does not normalize rewards')
        values = super().sample(n, norm_rews=False, device=device)
        indices = values[6].cpu()
        return ExecutionSample(values, self.enabled[indices].to(device), self.next_enabled[indices].to(device))

    def state_dict(self):
        raise NotImplementedError('repair v1 starts fresh; replay resume is not supported')

    def load_state_dict(self, state):
        raise NotImplementedError('repair replay cannot load historical replay without execution gates')
