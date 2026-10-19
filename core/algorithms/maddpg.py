import copy
import operator
from collections.abc import Mapping
from typing import Any
from typing import Optional
from typing import Union
import torch
import torch.nn.functional as F
from core.algorithms.misc import soft_update_group
from core.algorithms.misc import onehot_from_logits
from core.algorithms.misc import gumbel_softmax
from core.algorithms.agents import DDPGAgent
from core.algorithms.agents import validate_optimizer_state
from core.algorithms.agents import optimizer_transaction

class MADDPG(object):

    def __init__(self, agent_init_params, alg_types, gamma=0.95, tau=0.005, lr=0.01, lr_actor=None, lr_critic=None, hidden_dim=64, discrete_action=False, agent_role_names=None, agent_noise_sigmas=None, residual_action_reg=0.01):
        if not isinstance(alg_types, (list, tuple)) or not alg_types or (not isinstance(agent_init_params, (list, tuple))) or (len(agent_init_params) != len(alg_types)) or any((kind not in ('MADDPG', 'DDPG') for kind in alg_types)):
            raise ValueError('MADDPG requires matching nonempty agent definitions and algorithm types')
        self.nagents = len(alg_types)
        for (name, values) in (('roles', agent_role_names), ('noise sigmas', agent_noise_sigmas)):
            if values is not None and len(values) != self.nagents:
                raise ValueError('MADDPG agent count mismatch for ' + name)
        self.alg_types = alg_types
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.lr_actor = lr if lr_actor is None else lr_actor
        self.lr_critic = lr if lr_critic is None else lr_critic
        self.hidden_dim = hidden_dim
        self.discrete_action = discrete_action
        self.agent_role_names = agent_role_names or [f'agent_{i}' for i in range(self.nagents)]
        self.residual_action_reg = float(residual_action_reg)
        self.agents = [DDPGAgent(lr=lr, lr_actor=self.lr_actor, lr_critic=self.lr_critic, discrete_action=discrete_action, hidden_dim=hidden_dim, **params) for params in agent_init_params]
        if agent_noise_sigmas is not None:
            for (ag, sigma) in zip(self.agents, agent_noise_sigmas):
                ag.scale_noise(float(sigma), multiply=False)
        self.device = torch.device('cpu')
        self.niter = 0
        self.init_dict = None
        self.checkpoint_metadata = None
        self.last_residual_regularization_term = 0.0
        self.last_actor_updated = False
        self._cached_sample_key = None
        self._cached_sample = None

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]=None) -> torch.device:
        if device is None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(device, torch.device):
            return device
        device_str = str(device).lower()
        if device_str == 'gpu':
            device_str = 'cuda'
        if device_str.startswith('cuda') and (not torch.cuda.is_available()):
            return torch.device('cpu')
        return torch.device(device_str)

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for (k, v) in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=device, non_blocking=True)

    @staticmethod
    def _recursive_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().clone()
        if isinstance(obj, dict):
            return {k: MADDPG._recursive_to_cpu(v) for (k, v) in obj.items()}
        if isinstance(obj, list):
            return [MADDPG._recursive_to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple((MADDPG._recursive_to_cpu(v) for v in obj))
        return obj

    @staticmethod
    def _clone_agent_params_to_cpu(agent: DDPGAgent):
        return {'policy': MADDPG._recursive_to_cpu(agent.policy.state_dict()), 'target_policy': MADDPG._recursive_to_cpu(agent.target_policy.state_dict()), 'policy_opt': MADDPG._recursive_to_cpu(agent.policy_optimizer.state_dict()), 'critic1': MADDPG._recursive_to_cpu(agent.critic1.state_dict()), 'target_critic1': MADDPG._recursive_to_cpu(agent.target_critic1.state_dict()), 'critic1_opt': MADDPG._recursive_to_cpu(agent.critic1_optimizer.state_dict()), 'critic2': MADDPG._recursive_to_cpu(agent.critic2.state_dict()), 'target_critic2': MADDPG._recursive_to_cpu(agent.target_critic2.state_dict()), 'critic2_opt': MADDPG._recursive_to_cpu(agent.critic2_optimizer.state_dict())}

    def _move_all_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        for a in self.agents:
            a.policy.to(device)
            a.target_policy.to(device)
            a.critic1.to(device)
            a.target_critic1.to(device)
            a.critic2.to(device)
            a.target_critic2.to(device)
            self._move_optimizer_state(a.policy_optimizer, device)
            self._move_optimizer_state(a.critic1_optimizer, device)
            self._move_optimizer_state(a.critic2_optimizer, device)
            if hasattr(a, 'sync_noise_device'):
                a.sync_noise_device()
        self.device = device
        self._clear_sample_cache()

    def _clear_sample_cache(self):
        self._cached_sample_key = None
        self._cached_sample = None

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def reset_noise(self):
        for agent in self.agents:
            agent.noise.reset()

    def _ensure_obs_tensor(self, obs: Any) -> torch.Tensor:
        if torch.is_tensor(obs):
            tensor = obs.to(device=self.device, dtype=torch.float32, non_blocking=True)
        else:
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def step(self, observations, explore=False):
        if len(observations) != self.nagents:
            raise ValueError(f'expected {self.nagents} agent observations, got {len(observations)}')
        with torch.no_grad():
            proc_obs = [self._ensure_obs_tensor(obs) for obs in observations]
            if any((value.ndim != 2 for value in proc_obs)):
                raise ValueError('rollout observations require a [B, observation_dim] matrix')
            batch_size = proc_obs[0].shape[0]
            for (i, value) in enumerate(proc_obs):
                if value.shape != (batch_size, self.agent_init_params[i]['num_in_pol']) or batch_size < 1:
                    raise ValueError(f'invalid rollout observation shape for agent {i}')
                self._require_finite(value, 'rollout observations')
            return [a.step(obs, explore=explore) for (a, obs) in zip(self.agents, proc_obs)]

    def _unpack_sample(self, sample):
        if len(sample) != 8:
            raise ValueError(f'Chapter-3 replay samples must contain 8 items, got {len(sample)}')
        return sample

    def _to_device(self, tensors, dtype=torch.float32):
        out = []
        for x in tensors:
            if torch.is_tensor(x):
                out.append(x.to(device=self.device, dtype=dtype, non_blocking=True))
            else:
                out.append(torch.as_tensor(x, dtype=dtype, device=self.device))
        return out

    def _prepare_sample(self, sample):
        (obs, acs, rews, next_obs, dones, weights, indices, success_flags) = self._unpack_sample(sample)
        for (name, values) in (('observations', obs), ('actions', acs), ('rewards', rews), ('next observations', next_obs), ('dones', dones)):
            if len(values) != self.nagents:
                raise ValueError(f'{name} must contain {self.nagents} agents')
        obs = self._to_device(obs)
        acs = self._to_device(acs)
        rews = self._to_device(rews)
        next_obs = self._to_device(next_obs)
        dones = self._to_device(dones)
        if obs[0].ndim != 2 or obs[0].shape[0] < 1:
            raise ValueError('replay observations require a nonempty [B, observation_dim] batch')
        size = obs[0].shape[0]
        for (i, spec) in enumerate(self.agent_init_params):
            for (name, value, dim) in (('observations', obs[i], spec['num_in_pol']), ('next observations', next_obs[i], spec['num_in_pol']), ('actions', acs[i], spec['num_out_pol'])):
                if value.shape != (size, dim):
                    raise ValueError(f'{name} for agent {i} must have shape {(size, dim)}')
        for (name, values) in (('rewards', rews), ('dones', dones)):
            for value in values:
                if value.shape not in ((size,), (size, 1)):
                    raise ValueError(f'{name} must have shape [B] or [B, 1]; broadcasting is forbidden')
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        if weights.shape not in ((size,), (size, 1)):
            raise ValueError('replay weights must have shape [B] or [B, 1]')
        weights = weights.reshape(size, 1)
        index_values = torch.as_tensor(indices, device=self.device)
        flags = torch.as_tensor(success_flags, device=self.device)
        if index_values.shape != (size,) or index_values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('replay indices must be an integer [B] vector')
        if bool((index_values < 0).any()):
            raise ValueError('replay indices must be nonnegative')
        if flags.shape != (size,) or not bool(((flags == 0) | (flags == 1)).all()):
            raise ValueError('replay success flags must be a binary [B] vector')
        for (name, values) in (('observations', obs), ('actions', acs), ('rewards', rews), ('next observations', next_obs), ('dones', dones), ('weights', [weights])):
            for value in values:
                self._require_finite(value, name)
        if any((not bool(((d == 0) | (d == 1)).all()) for d in dones)):
            raise ValueError('replay done flags must be binary')
        if bool((weights < 0).any()):
            raise ValueError('replay weights must be nonnegative')
        if not bool((weights > 0).any()):
            raise ValueError('replay weights must include a positive weight')
        packed = {'obs': obs, 'acs': acs, 'rews': rews, 'next_obs': next_obs, 'dones': dones, 'weights': weights, 'indices': indices, 'success_flags': success_flags}
        self._augment_sample(packed, sample)
        with torch.no_grad():
            next_acs = self._build_target_actions(next_obs)
            next_acs = self._project_policy_actions(packed, next_acs, next_state=True)
            target_vf_in = torch.cat((*next_obs, *next_acs), dim=1)
        joint_vf_in = torch.cat((*obs, *acs), dim=1)
        packed.update(next_acs=next_acs, target_vf_in=target_vf_in, joint_vf_in=joint_vf_in)
        return packed

    @staticmethod
    def _require_finite(value, name):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'nonfinite {name}')

    def _augment_sample(self, batch, sample):
        """Optional execution metadata; never implicitly enlarge network inputs."""

    def _project_policy_actions(self, batch, actions, *, next_state=False):
        return actions

    def _actor_sample_weights(self, batch, agent_i):
        return None

    def _build_target_actions(self, next_obs):
        if self.discrete_action:
            return [onehot_from_logits(pi(no)) for (pi, no) in zip(self.target_policies, next_obs)]
        return [torch.clamp(pi(no), -1.0, 1.0) for (pi, no) in zip(self.target_policies, next_obs)]

    def _build_policy_action(self, agent_i, obs_i, explore_gumbel=True):
        ag = self.agents[agent_i]
        if self.discrete_action:
            logits = ag.policy(obs_i)
            act = gumbel_softmax(logits, hard=True) if explore_gumbel else onehot_from_logits(logits)
            return (logits, act)
        raw = torch.clamp(ag.policy(obs_i), -1.0, 1.0)
        return (raw, raw)

    def _compute_target_q(self, ag: DDPGAgent, target_vf_in, rew_i, done_i):
        with torch.no_grad():
            q1_next = ag.target_critic1(target_vf_in)
            q2_next = ag.target_critic2(target_vf_in)
            active = ~done_i.view(-1, 1).bool()
            self._require_finite(q1_next[active], 'nonterminal target critic 1')
            self._require_finite(q2_next[active], 'nonterminal target critic 2')
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rew_i.view(-1, 1) + torch.where(active, self.gamma * min_q_next, torch.zeros_like(min_q_next))
            self._require_finite(target_q, 'critic target before clipping')
            return torch.clamp(target_q, -10.0, 10.0)

    def _target_critic_input(self, batch, agent_i):
        if self.alg_types[agent_i] == 'MADDPG':
            return batch['target_vf_in']
        return torch.cat((batch['next_obs'][agent_i], batch['next_acs'][agent_i]), dim=1)

    def _critic_input(self, batch, agent_i):
        obs = batch['obs']
        acs = batch['acs']
        if self.alg_types[agent_i] == 'MADDPG':
            return batch['joint_vf_in'] if batch['joint_vf_in'] is not None else torch.cat((*obs, *acs), dim=1)
        return torch.cat((obs[agent_i], acs[agent_i]), dim=1)

    def _validate_agent_index(self, agent_i):
        dtype = getattr(agent_i, 'dtype', None)
        if isinstance(agent_i, bool) or dtype is torch.bool or getattr(dtype, 'kind', None) == 'b':
            raise ValueError('agent index must be an integer in range')
        try:
            index = operator.index(agent_i)
        except TypeError as exc:
            raise ValueError('agent index must be an integer in range') from exc
        if not 0 <= index < self.nagents:
            raise ValueError('agent index outside learner agent range')
        return index

    @staticmethod
    def _validate_update_optimizers(agent, *, actor):
        validate_optimizer_state(agent.critic1_optimizer)
        validate_optimizer_state(agent.critic2_optimizer)
        if actor:
            validate_optimizer_state(agent.policy_optimizer)

    def update(self, sample, agent_i, parallel=False, logger=None):
        agent_i = self._validate_agent_index(agent_i)
        self._validate_update_optimizers(self.agents[agent_i], actor=True)
        self.last_actor_updated = False
        batch = self._prepare_sample(sample)
        ag = self.agents[agent_i]
        target_q = self._compute_target_q(ag, self._target_critic_input(batch, agent_i), batch['rews'][agent_i], batch['dones'][agent_i])
        vf_in = self._critic_input(batch, agent_i)
        q1 = ag.critic1(vf_in)
        q2 = ag.critic2(vf_in)
        td_error = (target_q - q1).detach().squeeze(-1)
        loss1 = F.smooth_l1_loss(q1, target_q, reduction='none')
        loss2 = F.smooth_l1_loss(q2, target_q, reduction='none')
        td_loss = loss1 + loss2
        vf_loss = (td_loss * batch['weights']).mean()
        self._require_finite(vf_loss, 'critic loss')
        ag.critic1_optimizer.zero_grad(set_to_none=True)
        ag.critic2_optimizer.zero_grad(set_to_none=True)
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.critic1.parameters(), 0.5, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(ag.critic2.parameters(), 0.5, error_if_nonfinite=True)
        with optimizer_transaction(ag.critic1_optimizer, ag.critic2_optimizer):
            ag.critic1_optimizer.step()
            ag.critic2_optimizer.step()
        ag.policy_optimizer.zero_grad(set_to_none=True)
        actor_weights = self._actor_sample_weights(batch, agent_i)
        if actor_weights is not None:
            actor_weights = actor_weights.reshape(-1, 1)
            self._require_finite(actor_weights, 'actor sample weights')
            if bool((actor_weights < 0).any()):
                raise ValueError('actor sample weights must be nonnegative')
            if not bool(actor_weights.sum() > 0):
                self.last_residual_regularization_term = 0.0
                return (float(vf_loss.item()), 0.0, td_error)

        def actor_mean(value):
            if actor_weights is None:
                return value.mean()
            per_row = value.reshape(value.shape[0], -1).mean(dim=1, keepdim=True)
            return (per_row * actor_weights).sum() / actor_weights.sum()
        (curr_pol_out, curr_pol_act) = self._build_policy_action(agent_i, batch['obs'][agent_i], explore_gumbel=True)
        all_pol_acs = []
        for (idx, (pi, ob)) in enumerate(zip(self.policies, batch['obs'])):
            if idx == agent_i:
                all_pol_acs.append(curr_pol_act)
            else:
                with torch.no_grad():
                    if self.discrete_action:
                        all_pol_acs.append(onehot_from_logits(pi(ob)))
                    else:
                        all_pol_acs.append(torch.clamp(pi(ob), -1.0, 1.0))
        all_pol_acs = self._project_policy_actions(batch, all_pol_acs)
        vf_in_pol = torch.cat((*batch['obs'], *all_pol_acs), dim=1) if self.alg_types[agent_i] == 'MADDPG' else torch.cat((batch['obs'][agent_i], all_pol_acs[agent_i]), dim=1)
        q_pol = ag.critic1(vf_in_pol)
        pol_loss = -actor_mean(q_pol)
        if self.residual_action_reg > 0.0:
            residual_regularization = actor_mean(curr_pol_out ** 2) * self.residual_action_reg
            self.last_residual_regularization_term = float(residual_regularization.detach().item())
            pol_loss = pol_loss + residual_regularization
        else:
            self.last_residual_regularization_term = 0.0
        if self.discrete_action:
            probs = F.softmax(curr_pol_out, dim=-1)
            entropy = actor_mean(-(probs * torch.log(probs + 1e-08)).sum(dim=-1, keepdim=True))
            pol_loss = pol_loss - 0.002 * entropy
        self._require_finite(pol_loss, 'actor loss')
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.policy.parameters(), 0.2, error_if_nonfinite=True)
        with optimizer_transaction(ag.policy_optimizer):
            ag.policy_optimizer.step()
        self.last_actor_updated = True
        if logger is not None:
            logger.add_scalars(f'agent{agent_i}/loss', {'critic': float(vf_loss.item()), 'actor': float(pol_loss.item())}, self.niter)
        return (float(vf_loss.item()), float(pol_loss.item()), td_error)

    def update_all_targets(self, compute_diff=False):
        diffs = {'policy': [], 'critic1': [], 'critic2': []} if compute_diff else None
        pairs = [(getattr(a, target), getattr(a, source)) for a in self.agents for (target, source) in (('target_critic1', 'critic1'), ('target_critic2', 'critic2'), ('target_policy', 'policy'))]
        values = iter(soft_update_group(pairs, self.tau, return_diff=compute_diff))
        for a in self.agents:
            (diff_c1, diff_c2, diff_pol) = (next(values), next(values), next(values))
            if compute_diff:
                diffs['critic1'].append(diff_c1)
                diffs['critic2'].append(diff_c2)
                diffs['policy'].append(diff_pol)
        self._clear_sample_cache()
        self.niter += 1
        return diffs

    def scale_noise(self, factor: float, multiply: bool=True):
        for agent in self.agents:
            agent.scale_noise(factor, multiply=multiply)

    def prep_training(self, device='cuda'):
        target_device = self._resolve_device(device)
        self._move_all_to(target_device)
        for a in self.agents:
            a.policy.train()
            a.critic1.train()
            a.critic2.train()
            a.target_policy.train()
            a.target_critic1.train()
            a.target_critic2.train()
            if hasattr(a, 'sync_noise_device'):
                a.sync_noise_device()

    def prep_rollouts(self, device=None):
        if device is not None:
            target_device = self._resolve_device(device)
            self._move_all_to(target_device)
        for a in self.agents:
            a.policy.eval()
            if hasattr(a, 'sync_noise_device'):
                a.sync_noise_device()

    def training_state_dict(self):
        """Return an independent CPU snapshot, including nested optimizer state."""
        return {'init_dict': copy.deepcopy(self.init_dict), 'agent_params': [self._clone_agent_params_to_cpu(a) for a in self.agents], 'niter': int(self.niter)}

    def load_training_state_dict(self, state):
        if not isinstance(state, Mapping) or state.get('init_dict') != self.init_dict:
            raise ValueError('MADDPG initialization metadata mismatch during resume')
        params = copy.deepcopy(state.get('agent_params', []))
        if not isinstance(params, (list, tuple)) or len(params) != len(self.agents):
            raise ValueError('MADDPG agent count mismatch during resume')
        niter = state.get('niter', 0)
        if type(niter) is not int or niter < 0:
            raise ValueError('MADDPG update count must be a nonnegative integer')
        for (agent, agent_params) in zip(self.agents, params):
            agent.validate_params(agent_params)
        for (agent, agent_params) in zip(self.agents, params):
            agent._load_validated_params(agent_params)
        self.niter = niter
        self._clear_sample_cache()

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01, lr=0.01, lr_actor=None, lr_critic=None, hidden_dim=64, residual_action_reg=0.01):
        agent_init_params = []
        alg_types = ['MADDPG' for _ in range(env.num_agents)]
        obs_dims = [env.observation_space[f'agent_{i}'].shape[0] for i in range(env.num_agents)]
        ac_dims = [env.action_space[f'agent_{i}'].shape[0] for i in range(env.num_agents)]
        total_critic_in = sum(obs_dims) + sum(ac_dims)
        for i in range(env.num_agents):
            num_in_pol = obs_dims[i]
            num_out_pol = ac_dims[i]
            num_in_critic = total_critic_in if alg_types[i] == 'MADDPG' else obs_dims[i] + ac_dims[i]
            params = {'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic}
            agent_init_params.append(params)
        role_names = getattr(env, 'role_names', [f'agent_{i}' for i in range(env.num_agents)])
        if hasattr(env, 'agent_specs') and len(env.agent_specs) == env.num_agents:
            noise_sigmas = []
            for spec in env.agent_specs:
                name = spec.get('name', '')
                if name == 'search_fast':
                    noise_sigmas.append(0.18)
                elif name == 'search_balanced':
                    noise_sigmas.append(0.14)
                elif name == 'search_precise':
                    noise_sigmas.append(0.1)
                elif name == 'executor':
                    noise_sigmas.append(0.08)
                else:
                    noise_sigmas.append(0.12)
        else:
            noise_sigmas = None
        effective_lr_actor = lr if lr_actor is None else lr_actor
        effective_lr_critic = lr if lr_critic is None else lr_critic
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr, 'lr_actor': effective_lr_actor, 'lr_critic': effective_lr_critic, 'hidden_dim': hidden_dim, 'alg_types': alg_types, 'agent_init_params': agent_init_params, 'discrete_action': False, 'agent_role_names': role_names, 'agent_noise_sigmas': noise_sigmas, 'residual_action_reg': float(residual_action_reg)}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
