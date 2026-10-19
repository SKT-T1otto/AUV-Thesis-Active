from torch.optim import Adam
import torch
import copy
import math
from collections.abc import Mapping
from contextlib import contextmanager
from core.algorithms.networks import MLPNetwork
from core.algorithms.misc import hard_update
from core.algorithms.misc import gumbel_softmax
from core.algorithms.misc import onehot_from_logits
from core.algorithms.noise import OUNoise

def _validate_adam_slot(slot, parameter, *, amsgrad):
    required = {'step', 'exp_avg', 'exp_avg_sq'}
    if amsgrad:
        required.add('max_exp_avg_sq')
    if not isinstance(slot, Mapping) or set(slot) != required:
        raise ValueError('invalid Adam state fields')
    step = slot['step']
    if torch.is_tensor(step):
        if step.numel() != 1 or step.is_complex() or step.dtype == torch.bool:
            raise ValueError('Adam step must be a finite nonnegative integer')
        step = step.item()
    if isinstance(step, bool) or not isinstance(step, (int, float)) or (not math.isfinite(step)) or (step < 0) or (step != int(step)):
        raise ValueError('Adam step must be a finite nonnegative integer')
    for name in required - {'step'}:
        value = slot[name]
        if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype or (value.layout != torch.strided) or (not bool(torch.isfinite(value).all())):
            raise ValueError('invalid Adam moment: ' + name)
        if name != 'exp_avg' and bool((value < 0).any()):
            raise ValueError('Adam second moment must be nonnegative: ' + name)
    if amsgrad and bool((slot['max_exp_avg_sq'] < slot['exp_avg_sq']).any()):
        raise ValueError('Adam maximum second moment is smaller than second moment')

def validate_optimizer_state(optimizer):
    """Check Adam invariants before either twin optimizer can write parameters."""
    if not isinstance(optimizer, Adam):
        return
    parameters = {p for group in optimizer.param_groups for p in group['params']}
    if set(optimizer.state) - parameters:
        raise ValueError('Adam state refers to an unknown parameter')
    for group in optimizer.param_groups:
        for name in ('lr', 'eps', 'weight_decay'):
            value = group[name]
            if torch.is_tensor(value):
                value = value.item()
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid Adam hyperparameter: ' + name)
        if len(group['betas']) != 2 or any((not math.isfinite(b) or not 0 <= b < 1 for b in group['betas'])):
            raise ValueError('invalid Adam betas')
        for parameter in group['params']:
            slot = optimizer.state.get(parameter)
            if slot:
                _validate_adam_slot(slot, parameter, amsgrad=group['amsgrad'])

def validate_module_snapshot(module, incoming, name='network'):
    expected = module.state_dict()
    if not isinstance(incoming, Mapping) or set(incoming) != set(expected):
        raise ValueError('network state keys mismatch: ' + name)
    for (key, reference) in expected.items():
        value = incoming[key]
        if not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype or (value.layout != torch.strided) or (not bool(torch.isfinite(value).all())):
            raise ValueError('invalid network state: ' + name + '.' + key)

def validate_optimizer_snapshot(optimizer, incoming, name='optimizer'):
    expected = optimizer.state_dict()
    if not isinstance(incoming, Mapping) or set(incoming) != {'state', 'param_groups'}:
        raise ValueError('invalid optimizer state: ' + name)
    groups = incoming['param_groups']
    if not isinstance(groups, (list, tuple)) or len(groups) != len(expected['param_groups']):
        raise ValueError('optimizer group count mismatch: ' + name)
    identifiers = []
    for (group, reference, live) in zip(groups, expected['param_groups'], optimizer.param_groups):
        if not isinstance(group, Mapping) or set(group) != set(reference) or {k: v for (k, v) in group.items() if k != 'params'} != {k: v for (k, v) in reference.items() if k != 'params'} or (not isinstance(group['params'], (list, tuple))) or (len(group['params']) != len(live['params'])):
            raise ValueError('optimizer configuration mismatch: ' + name)
        identifiers.extend(group['params'])
    if any((type(i) is not int or i < 0 for i in identifiers)) or len(set(identifiers)) != len(identifiers):
        raise ValueError('invalid optimizer parameter identifiers: ' + name)
    if not isinstance(incoming['state'], Mapping) or set(incoming['state']) - set(identifiers):
        raise ValueError('unknown optimizer parameter: ' + name)
    for (group, live) in zip(groups, optimizer.param_groups):
        for (identifier, parameter) in zip(group['params'], live['params']):
            if identifier in incoming['state'] and isinstance(optimizer, Adam):
                _validate_adam_slot(incoming['state'][identifier], parameter, amsgrad=group['amsgrad'])
    staged = copy.deepcopy(optimizer)
    staged.load_state_dict(copy.deepcopy(incoming))
    validate_optimizer_state(staged)

@contextmanager
def optimizer_transaction(*optimizers):
    """Commit a group of steps only with finite parameters and valid state.

    Computation/gradient checks precede this block. On a write failure restore
    both model parameters and optimizer buffers, then propagate the exception.
    """
    parameters = list(dict.fromkeys((p for opt in optimizers for g in opt.param_groups for p in g['params'])))
    for opt in optimizers:
        validate_optimizer_state(opt)
    if any((not bool(torch.isfinite(p).all()) for p in parameters)):
        raise FloatingPointError('nonfinite parameter before optimizer step')
    saved_parameters = [p.detach().clone() for p in parameters]
    saved_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
    try:
        yield
        if any((not bool(torch.isfinite(p).all()) for p in parameters)):
            raise FloatingPointError('optimizer wrote nonfinite parameters')
        for opt in optimizers:
            validate_optimizer_state(opt)
    except Exception:
        with torch.no_grad():
            for (p, saved) in zip(parameters, saved_parameters):
                p.copy_(saved)
        for (opt, saved) in zip(optimizers, saved_states):
            opt.load_state_dict(saved)
        raise

class DDPGAgent(object):

    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=128, lr=0.0003, lr_critic=0.0005, lr_actor=0.001, discrete_action=False):
        actor_hidden = int(hidden_dim)
        critic_hidden = int(max(hidden_dim * 2, hidden_dim))
        self.policy = MLPNetwork(input_dim=num_in_pol, out_dim=num_out_pol, hidden_dim=actor_hidden, constrain_out=True, discrete_action=discrete_action, norm_in=True)
        self.target_policy = MLPNetwork(input_dim=num_in_pol, out_dim=num_out_pol, hidden_dim=actor_hidden, constrain_out=True, discrete_action=discrete_action, norm_in=True)
        self.critic1 = MLPNetwork(input_dim=num_in_critic, out_dim=1, hidden_dim=critic_hidden, constrain_out=False, discrete_action=discrete_action, norm_in=True)
        self.target_critic1 = MLPNetwork(input_dim=num_in_critic, out_dim=1, hidden_dim=critic_hidden, constrain_out=False, discrete_action=discrete_action, norm_in=True)
        self.critic2 = MLPNetwork(input_dim=num_in_critic, out_dim=1, hidden_dim=critic_hidden, constrain_out=False, discrete_action=discrete_action, norm_in=True)
        self.target_critic2 = MLPNetwork(input_dim=num_in_critic, out_dim=1, hidden_dim=critic_hidden, constrain_out=False, discrete_action=discrete_action, norm_in=True)
        self.initial_sigma = 0.2
        self.min_sigma = 0.02
        self.noise_decay = 2e-05
        self.current_sigma = self.initial_sigma
        self.noise = OUNoise(num_out_pol, sigma=self.initial_sigma)
        self.discrete_action = discrete_action
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic1, self.critic1)
        hard_update(self.target_critic2, self.critic2)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr_critic)

    def _policy_device(self):
        return next(self.policy.parameters()).device

    def sync_noise_device(self):
        self.noise.to(device=self._policy_device())

    def step(self, obs, explore=False):
        self.sync_noise_device()
        with torch.no_grad():
            action = self.policy(obs)
            if explore:
                noise = self.noise.sample().to(device=action.device, dtype=action.dtype)
                if action.ndim > 1:
                    noise = noise.unsqueeze(0).expand_as(action)
                action = torch.clamp(action + noise, -1.0, 1.0)
            if self.discrete_action:
                action = gumbel_softmax(action, hard=True) if explore else onehot_from_logits(action)
        return action

    def validate_params(self, params):
        """Preflight the entire agent without changing modules or optimizers."""
        modules = ('policy', 'target_policy', 'critic1', 'target_critic1', 'critic2', 'target_critic2')
        optimizers = {'policy_opt': self.policy_optimizer, 'critic1_opt': self.critic1_optimizer, 'critic2_opt': self.critic2_optimizer}
        if not isinstance(params, Mapping) or set(params) != set(modules) | set(optimizers):
            raise ValueError('invalid DDPG agent state fields')
        for name in modules:
            validate_module_snapshot(getattr(self, name), params[name], name)
        for (name, optimizer) in optimizers.items():
            validate_optimizer_snapshot(optimizer, params[name], name)

    def _load_validated_params(self, params):
        """Commit only after every agent in the learner has passed preflight."""
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_opt'])
        self.critic1.load_state_dict(params['critic1'])
        self.target_critic1.load_state_dict(params['target_critic1'])
        self.critic1_optimizer.load_state_dict(params['critic1_opt'])
        self.critic2.load_state_dict(params['critic2'])
        self.target_critic2.load_state_dict(params['target_critic2'])
        self.critic2_optimizer.load_state_dict(params['critic2_opt'])
        self.sync_noise_device()

    def scale_noise(self, sigma: float, multiply: bool=False):
        if multiply:
            self.noise.sigma *= sigma
        else:
            self.noise.sigma = sigma
        self.current_sigma = float(self.noise.sigma)

    @property
    def critic(self):
        return self.critic1

    @critic.setter
    def critic(self, val):
        self.critic1 = val
