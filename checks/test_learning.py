"""Current B2/B3 numerical and interface regression tests, fresh models only."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
root = Path(__file__).resolve().parents[1]
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
from chapter3_bser.learning.model import build_model
from chapter3_bser.learning.replay import ExecutionReplay
from chapter3_bser.learning.replay_base import ExecutionSample
from chapter3_bser.learning.boundary import boundary_features
from chapter3_bser.learning.checkpoint import state_digest
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.rewards.team_objective import RewardAccounting
from core.replay.ch3_buffer import CH3ReplayBuffer

CONFIGS={k:json.loads((root/f'configs/{k}.json').read_text(encoding='utf-8')) for k in ('b2','b3')}

def fresh(label='b2'):
    torch.manual_seed(927391)
    return build_model(CONFIGS[label])

def transition(marker=0,*,terminal=False,frozen=False):
    gen=torch.Generator().manual_seed(82729+marker)
    obs=torch.rand(4,28,generator=gen);nxt=torch.rand(4,28,generator=gen)
    obs[:,20]=0;nxt[:,20]=0
    enabled=torch.ones(4,dtype=torch.bool);next_enabled=enabled.clone()
    actions=torch.rand(4,3,generator=gen)*.4-.2
    if frozen:
        obs[:,20]=1;nxt[:,20]=1;enabled.zero_();next_enabled.zero_();actions.zero_()
    if terminal:next_enabled.zero_()
    return dict(observations=obs,next_observations=nxt,actions=actions,rewards=torch.full((4,),float(marker)/10),
                dones=torch.full((4,),terminal),action_enabled=enabled,next_action_enabled=next_enabled,
                success_flags=torch.zeros(4,dtype=torch.bool))

def sample(*,frozen=False):
    replay=ExecutionReplay(4)
    for i in range(4):replay.push_transition(transition(i,frozen=frozen))
    return replay.sample(4)

def modified(s,index,value):
    values=list(s);values[index]=value
    return ExecutionSample(values,s.enabled.clone(),s.next_enabled.clone())

def snap(model):return state_digest(model.training_state_dict())

def params(module):return {k:v.detach().clone() for k,v in module.state_dict().items()}

def same(a,b):return all(torch.equal(a[k],b[k]) for k in a) and a.keys()==b.keys()

class Constants(nn.Module):
    def __init__(self,values):super().__init__();self.values=values
    def forward(self,x):return torch.as_tensor(self.values,dtype=x.dtype,device=x.device).view(-1,1)

class ActiveRLTests(unittest.TestCase):
    def test_01_network_dimensions_ownership_and_initial_targets(self):
        for label in CONFIGS:
            with self.subTest(label=label):
                model=fresh(label)
                self.assertEqual(model.nagents,4);self.assertEqual(model.gamma,.95)
                for a in model.agents:
                    actor=a.policy.actor if label=='b3' else a.policy
                    self.assertEqual((actor.fc1.in_features,actor.fc3.out_features),(28,3))
                    self.assertEqual((a.critic1.fc1.in_features,a.critic1.fc3.out_features),(124,1))
                    self.assertEqual(actor.fc1.out_features,128);self.assertEqual(a.critic1.fc1.out_features,256)
                    self.assertTrue(same(params(a.policy),params(a.target_policy)))
                    self.assertTrue(same(params(a.critic1),params(a.target_critic1)))
                    self.assertTrue(same(params(a.critic2),params(a.target_critic2)))
                    self.assertEqual({id(p) for g in a.policy_optimizer.param_groups for p in g['params']},{id(p) for p in a.policy.parameters()})
                    self.assertEqual(a.policy_optimizer.param_groups[0]['lr'],.001)
                    self.assertEqual(a.critic1_optimizer.param_groups[0]['lr'],.0005)
                    self.assertFalse({id(p) for p in a.policy.parameters()} & {id(p) for p in a.critic1.parameters()})

    def test_02_terminal_nan_and_independent_bellman_oracle(self):
        m=fresh();a=m.agents[0]
        a.target_critic1=Constants([float('nan'),2,100,-100])
        a.target_critic2=Constants([float('nan'),3,90,-90])
        rewards=torch.tensor([1.,.3,1.,-1.]);dones=torch.tensor([1.,0,0,0])
        actual=m._compute_target_q(a,torch.zeros(4,124),rewards,dones)
        expected=torch.tensor([[1.],[.3+.95*2],[10.],[-10.]])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        a.target_critic1=Constants([float('nan')]*4)
        with self.assertRaises(FloatingPointError):m._compute_target_q(a,torch.zeros(4,124),rewards,torch.zeros(4))

    def test_03_batch_shape_broadcast_and_nonfinite_rejected_without_update(self):
        for label in CONFIGS:
            with self.subTest(label=label):
                m=fresh(label);s=sample();before=snap(m)
                for invalid in (torch.zeros(3),torch.zeros(4,2),torch.full((4,),float('nan'))):
                    rewards=list(s[2]);rewards[0]=invalid
                    with self.assertRaises((ValueError,FloatingPointError)):m.update(modified(s,2,rewards),0)
                    self.assertEqual(before,snap(m))
                weights=torch.ones(4,4)
                with self.assertRaises(ValueError):m.update(modified(s,5,weights),0)
                self.assertEqual(before,snap(m))

    def test_04_frozen_actor_keeps_existing_adam_but_critic_learns(self):
        for label in CONFIGS:
            with self.subTest(label=label):
                m=fresh(label);m.update(sample(),0);m.update_all_targets()
                a=m.agents[0];before=params(a.policy);adam=state_digest(a.policy_optimizer.state_dict());critic=params(a.critic1)
                m.update(sample(frozen=True),0)
                self.assertFalse(m.last_actor_updated);self.assertTrue(same(before,params(a.policy)))
                self.assertEqual(adam,state_digest(a.policy_optimizer.state_dict()))
                self.assertFalse(same(critic,params(a.critic1)));self.assertEqual(m.actor_updates[0],1)

    def test_05_one_agent_update_does_not_write_other_agents_or_targets(self):
        for label in CONFIGS:
            with self.subTest(label=label):
                m=fresh(label);before=[state_digest(m._clone_agent_params_to_cpu(a)) for a in m.agents]
                target=params(m.agents[0].target_policy);actor=params(m.agents[0].policy)
                if label=='b3':encoder=params(m.agents[0].policy.boundary_encoder)
                m.update(sample(),0)
                self.assertTrue(m.last_actor_updated);self.assertFalse(same(actor,params(m.agents[0].policy)))
                self.assertTrue(same(target,params(m.agents[0].target_policy)))
                for i in (1,2,3):self.assertEqual(before[i],state_digest(m._clone_agent_params_to_cpu(m.agents[i])))
                if label=='b3':self.assertFalse(same(encoder,params(m.agents[0].policy.boundary_encoder)))

    def test_06_soft_update_uses_tau_and_is_atomic_on_late_invalid_source(self):
        m=fresh();m.update(sample(),0)
        before=[[p.clone() for p in a.target_policy.parameters()] for a in m.agents]
        source=[[p.clone() for p in a.policy.parameters()] for a in m.agents]
        m.update_all_targets()
        for i,a in enumerate(m.agents):
            for p,old,src in zip(a.target_policy.parameters(),before[i],source[i]):
                torch.testing.assert_close(p,old.lerp(src,.005),rtol=0,atol=0)
        before=[params(a.target_policy) for a in m.agents]
        with torch.no_grad():next(m.agents[-1].policy.parameters()).flatten()[0]=float('nan')
        with self.assertRaises(FloatingPointError):m.update_all_targets()
        for a,old in zip(m.agents,before):self.assertTrue(same(params(a.target_policy),old))
        self.assertEqual(m.niter,1)

    def test_07_both_critics_unchanged_on_second_critic_bad_gradient(self):
        m=fresh();s=sample();before=snap(m)
        hook=next(m.agents[0].critic2.parameters()).register_hook(lambda g:torch.full_like(g,float('nan')))
        try:
            with self.assertRaises(RuntimeError):m.update(s,0)
        finally:hook.remove()
        self.assertEqual(before,snap(m))

    def test_08_twin_optimizer_rollback_if_second_step_throws(self):
        m=fresh();before=snap(m)
        with patch.object(m.agents[0].critic2_optimizer,'step',side_effect=RuntimeError('injected second optimizer failure')):
            with self.assertRaisesRegex(RuntimeError,'injected'):m.update(sample(),0)
        self.assertEqual(before,snap(m))

    def test_09_b3_next_and_current_context_rebound_from_each_batch(self):
        m=fresh('b3');s=sample()
        m.step([torch.zeros(28) for _ in range(4)])
        batch=m._prepare_sample(s)
        current=boundary_features(torch.stack(s[0],dim=1));nxt=boundary_features(torch.stack(s[3],dim=1))
        for i,a in enumerate(m.agents):
            torch.testing.assert_close(a.policy._boundary_context,current[:,i],rtol=0,atol=0)
            torch.testing.assert_close(a.target_policy._boundary_context,nxt[:,i],rtol=0,atol=0)
            with torch.no_grad():
                expected=a.target_policy.actor(a.target_policy.boundary_encoder(s[3][i],nxt[:,i])).clamp(-1,1)*s.next_enabled[:,i:i+1]
            torch.testing.assert_close(batch['next_acs'][i],expected,rtol=0,atol=0)
        other=sample(frozen=True);m._prepare_sample(other)
        expected=boundary_features(torch.stack(other[0],dim=1))
        for i,a in enumerate(m.agents):torch.testing.assert_close(a.policy._boundary_context,expected[:,i],rtol=0,atol=0)

    def test_10_gate_projection_and_team_terminal_contract(self):
        for label in CONFIGS:
            with self.subTest(label=label):
                m=fresh(label);s=sample();s.enabled[:,0]=False
                values=list(s);values[1]=[v.clone() for v in s[1]];values[1][0].zero_()
                s=ExecutionSample(values,s.enabled,s.next_enabled)
                batch=m._prepare_sample(s)
                out=m._project_policy_actions(batch,[torch.ones(4,3) for _ in range(4)])
                self.assertEqual(out[0].abs().sum().item(),0);self.assertEqual(out[1].sum().item(),12)
                s.next_enabled[:,2]=False;batch=m._prepare_sample(s);self.assertEqual(batch['next_acs'][2].abs().sum().item(),0)
                done=[v.clone() for v in s[4]];done[0][0]=1
                with self.assertRaisesRegex(ValueError,'mission dones'):m._prepare_sample(modified(s,4,done))

    def test_11_replay_wrap_and_gate_rows_match_exact_transition(self):
        r=ExecutionReplay(3)
        expected={}
        for i in range(5):
            t=transition(i,terminal=(i%2==0));slot=r.next_idx;expected[slot]=copy.deepcopy(t);r.push_transition(t)
        self.assertEqual((len(r),r.next_idx),(3,2));s=r.sample(3)
        for j,idx in enumerate(s[6].tolist()):
            t=expected[idx]
            for a in range(4):
                torch.testing.assert_close(s[0][a][j],t['observations'][a],rtol=0,atol=0)
                torch.testing.assert_close(s[1][a][j],t['actions'][a],rtol=0,atol=0)
                torch.testing.assert_close(s[2][a][j],t['rewards'][a],rtol=0,atol=0)
                torch.testing.assert_close(s[3][a][j],t['next_observations'][a],rtol=0,atol=0)
            self.assertTrue(torch.equal(s.enabled[j],t['action_enabled']))
            self.assertTrue(torch.equal(s.next_enabled[j],t['next_action_enabled']))
        torch.testing.assert_close(s[5],torch.ones(3),rtol=0,atol=0)

    def test_12_bad_replay_write_is_rejected_without_ring_mutation(self):
        r=ExecutionReplay(2);r.push_transition(transition(0));before=r.obs_buffs[0].clone();index=r.next_idx
        t=transition(1);t['action_enabled'][0]=False
        with self.assertRaises(ValueError):r.push_transition(t)
        self.assertEqual(r.next_idx,index);self.assertEqual(len(r),1)
        self.assertTrue(torch.equal(before[:1],r.obs_buffs[0][:1]))
        t=transition(1);t['dones'][0]=True
        with self.assertRaises(ValueError):r.push_transition(t)
        self.assertEqual(r.next_idx,index)
        with self.assertRaises(TypeError):r.push(*([None]*6))
        with self.assertRaises(ValueError):r.sample(1,norm_rews=True)

    def test_13_snapshot_is_deep_and_invalid_load_is_atomic(self):
        for label in CONFIGS:
            with self.subTest(label=label):
                m=fresh(label);m.update(sample(),0);m.update_all_targets();saved=m.training_state_dict();old=state_digest(saved)
                m.update(sample(),0)
                self.assertEqual(old,state_digest(saved))
                clone=fresh(label);clone.load_training_state_dict(saved);self.assertEqual(snap(clone),old)
                bad=copy.deepcopy(saved);k=next(iter(bad['agent_params'][-1]['policy']))
                bad['agent_params'][-1]['policy'][k]=torch.zeros(1)
                before=snap(clone)
                with self.assertRaises(ValueError):clone.load_training_state_dict(bad)
                self.assertEqual(before,snap(clone))

    def test_14_sample_changes_are_not_hidden_by_cached_identity(self):
        m=fresh();s=sample();a=m._prepare_sample(s)['target_vf_in'].clone()
        s[3][0][:,:3].add_(.5);b=m._prepare_sample(s)['target_vf_in']
        self.assertFalse(torch.equal(a,b))

    def test_15_core_replay_restore_checks_ring_relationship(self):
        r=CH3ReplayBuffer(4,4,[28]*4,[3]*4);t=transition()
        r.push(t['observations'],t['actions'],t['rewards'],t['next_observations'],t['dones'],t['success_flags'])
        saved=r.state_dict();bad=copy.deepcopy(saved);bad['next_idx']=0
        with self.assertRaises(ValueError):r.load_state_dict(bad)
        self.assertEqual(state_digest(r.state_dict()),state_digest(saved))

    def test_16_team_reward_uses_immediate_mean_not_discounted_reward(self):
        accounting=RewardAccounting(dict(reward_objective='team_mean_v1'),gamma=.95)
        first=accounting.apply(torch.tensor([1.,2.,3.,4.]),{})
        second=accounting.apply(torch.tensor([2.,2.,2.,2.]),{})
        torch.testing.assert_close(first,torch.full((4,),2.5),rtol=0,atol=0)
        torch.testing.assert_close(second,torch.full((4,),2.),rtol=0,atol=0)
        self.assertAlmostEqual(accounting.team_discounted_return,2.5+.95*2)
        with self.assertRaises(RuntimeError):accounting.apply(first,{'reward_transform_applied_count':1})

    def test_17_physical_fingerprint_rejects_id_only_relabel(self):
        s=json.loads((root/'verification/mechanism_scenes.json').read_text(encoding='utf-8'))['scenarios'][0]
        renamed=copy.deepcopy(s);renamed.update(scenario_id='renamed',scenario_seed=999999,planner_seed=123,scenario_role='eval',scenario_split='eval')
        self.assertEqual(scenario_fingerprint(s),scenario_fingerprint(renamed))
        moved=copy.deepcopy(s);moved['initial_agent_positions'][0][0]+=.1
        self.assertNotEqual(scenario_fingerprint(s),scenario_fingerprint(moved))

