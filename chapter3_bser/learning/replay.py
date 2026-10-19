import torch
from chapter3_bser.learning.replay_base import ExecutionReplay as ParentReplay

class ExecutionReplay(ParentReplay):

    def __init__(self, capacity):
        super().__init__(capacity)
        self.enabled.zero_()
        self.next_enabled.zero_()
        self.gate_initialized = torch.zeros(capacity, dtype=torch.bool)

    def push(self, *args, **kwargs):
        raise TypeError('execution replay requires push_transition with current and next gates')

    def push_transition(self, transition):
        prepared = {}
        shapes = dict(observations=(4, 28), next_observations=(4, 28), actions=(4, 3), rewards=(4,), dones=(4,), action_enabled=(4,), next_action_enabled=(4,))
        for (name, shape) in shapes.items():
            if name not in transition:
                raise ValueError(f'missing transition field: {name}')
            value = torch.as_tensor(transition[name]).detach().cpu().clone()
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f'invalid transition {name} shape or finite values')
            if name in ('dones', 'action_enabled', 'next_action_enabled'):
                if not bool(((value == 0) | (value == 1)).all()):
                    raise ValueError(f'{name} must be binary')
                value = value.bool()
            else:
                value = value.float()
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f'{name} overflows float32')
            prepared[name] = value
        success = torch.as_tensor(transition.get('success_flags', [False] * 4)).detach().cpu()
        if success.shape != (4,) or not bool(((success == 0) | (success == 1)).all()):
            raise ValueError('success flags must be four binary values')
        prepared['success_flags'] = success.bool().clone()
        if bool((prepared['actions'].abs() > 1).any()) or bool((prepared['actions'][~prepared['action_enabled']] != 0).any()):
            raise ValueError('actions must be applied normalized residuals, zero when disabled')
        if not bool((prepared['dones'] == prepared['dones'][0]).all()):
            raise ValueError('searcher freeze is not a local mission done')
        if bool(prepared['dones'][0]) and bool(prepared['next_action_enabled'].any()):
            raise ValueError('terminal next-state gates must be zero')
        index = self.next_idx
        super().push_transition(prepared)
        self.gate_initialized[index] = True

    def sample(self, n, *, norm_rews=False, device='cpu'):
        if not bool(self.gate_initialized[:self.filled_i].all()):
            raise ValueError('execution replay contains an uninitialized gate slot')
        for value in (self.enabled, self.next_enabled):
            if value.shape != (self.max_steps, 4) or value.dtype != torch.bool:
                raise ValueError('execution gate storage was corrupted')
        sample = super().sample(n, norm_rews=norm_rews, device=device)
        indices = sample[6].cpu()
        if indices.dtype != torch.int64 or bool((indices < 0).any()) or bool((indices >= self.filled_i).any()) or (not bool(self.gate_initialized[indices].all())):
            raise ValueError('sample refers to an unwritten execution replay slot')
        return sample
