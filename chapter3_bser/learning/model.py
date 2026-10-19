import copy
import torch
from core.algorithms.maddpg import MADDPG
from chapter3_bser.learning.execution_mask import ExecutionRepair as ParentRepair
from chapter3_bser.learning.boundary import _contract_environment
from chapter3_bser.learning.boundary import BoundaryConditionedMADDPG
from chapter3_bser.learning.boundary import BoundaryConditionedPolicy
from chapter3_bser.learning.config import VERSION
from chapter3_bser.learning.config import validate

class ExecutionRepair(ParentRepair):

    def _augment_sample(self, batch, sample):
        super()._augment_sample(batch, sample)
        terminal = batch['dones'][0].reshape(-1).bool()
        if any((not torch.equal(d.reshape(-1).bool(), terminal) for d in batch['dones'])):
            raise ValueError('mission dones must be shared, not searcher freeze flags')
        if bool((batch['next_enabled'][terminal] != 0).any()):
            raise ValueError('terminal next gates must be zero')
        for (i, action) in enumerate(batch['acs']):
            if bool((action.abs() > 1).any()):
                raise ValueError('execution replay actions must be normalized')
            if self.repairs['inactive_actor']:
                for (field, obs) in (('enabled', batch['obs']), ('next_enabled', batch['next_obs'])):
                    if bool(((batch[field][:, i] != 0) & (obs[i][:, 20] >= 0.5)).any()):
                        raise ValueError('execution gate contradicts a frozen agent')

    def training_state_dict(self):
        return dict(**MADDPG.training_state_dict(self), repair_version=VERSION, repairs=copy.deepcopy(self.repairs), actor_updates=list(self.actor_updates))

    def load_training_state_dict(self, state):
        if state.get('repair_version') != VERSION or state.get('repairs') != self.repairs:
            raise ValueError('old or different repair weights cannot be relabeled as v2')
        counts = state.get('actor_updates', [])
        if len(counts) != 4 or any((type(n) is not int or not 0 <= n <= state['niter'] for n in counts)):
            raise ValueError('invalid actor update counts')
        MADDPG.load_training_state_dict(self, state)
        self.actor_updates = list(counts)

class RepairedMADDPG(ExecutionRepair, MADDPG):
    pass

class RepairedBoundaryMADDPG(ExecutionRepair, BoundaryConditionedMADDPG):
    pass

def build_model(config, device='cpu'):
    c = validate(config)
    rl = c['rl']
    boundary = c['algorithm'] == 'direct_boundary_maddpg'
    cls = RepairedBoundaryMADDPG if boundary else RepairedMADDPG
    model = cls.init_from_env(_contract_environment(), gamma=rl['gamma'], tau=rl['tau'], lr_actor=rl['lr_actor'], lr_critic=rl['lr_critic'], hidden_dim=rl['hidden_dim'], residual_action_reg=0.0)
    if boundary:
        for agent in model.agents:
            agent.policy = BoundaryConditionedPolicy(agent.policy, c.get('boundary_encoder', {}).get('hidden_dim', 32))
            agent.target_policy = copy.deepcopy(agent.policy)
            agent.policy_optimizer = torch.optim.Adam(agent.policy.parameters(), lr=model.lr_actor)
    model.repairs = copy.deepcopy(c['repairs'])
    model.actor_updates = [0] * 4
    model.last_target_diagnostics = {}
    model.prep_training(device=device)
    return model
