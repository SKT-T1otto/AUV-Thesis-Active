"""Separate finite-horizon factual-risk learning and recovery-action learning.

No task rewards/network/optimizer are changed. Unobserved horizons are censored.
Risk estimates concern the recorded behavior mixture, not a proven probability
bound for arbitrary alternative continuation policies.
"""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn
from chapter3_bser.safe_rl.options import FRAME_DIM


def horizon_labels(costs,horizon):
    costs=np.asarray(costs,bool)
    if costs.ndim!=2 or costs.shape[1]!=4:raise ValueError('per-agent collision costs required')
    n=len(costs);labels=np.zeros((n,4,horizon),np.float32);mask=np.zeros_like(labels)
    for t in range(n):
        for i in range(4):
            future=costs[t:min(n,t+horizon),i];hits=np.flatnonzero(future)
            observed=min(horizon,n-t);mask[t,i,:observed]=1
            if len(hits):
                first=int(hits[0]);labels[t,i,first:]=1;mask[t,i,:]=1
    return labels,mask


class SafetyLearner:
    def __init__(self,options,*,seed=None):
        self.options=copy.deepcopy(options);self.updates=0;self.input_dim=FRAME_DIM*options['history']
        self.rng=np.random.default_rng(options['seed'] if seed is None else seed)
        # Isolated initialization preserves the original task policy's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options['seed'] if seed is None else seed)
            h=options['hidden_dim']
            self.actor=nn.Sequential(nn.Linear(self.input_dim,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,3),nn.Tanh())
            self.critics=nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim+3,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,options['horizon'])) for _ in range(2)])
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=options['actor_lr'])
        self.critic_optimizer=torch.optim.Adam(self.critics.parameters(),lr=options['critic_lr'])
        self.replay=deque(maxlen=options['replay_size'])

    @property
    def ready(self):return self.updates>=self.options['minimum_updates']

    def action(self,features):
        with torch.no_grad():return self.actor(torch.as_tensor(features,dtype=torch.float32)).numpy().copy()

    def risk(self,features,action):
        with torch.no_grad():
            x=torch.cat((torch.as_tensor(features,dtype=torch.float32),torch.as_tensor(action,dtype=torch.float32)))
            values=torch.stack([torch.sigmoid(net(x)[-1]) for net in self.critics]).numpy()
        return dict(score=float(values.max()),mean=float(values.mean()),disagreement=float(values.std()),
            horizon=self.options['horizon'],calibrated_probability_bound=False,continuation='recorded_behavior_mixture')

    def add_episode(self,transitions):
        costs=np.array([x['collision_cost'] for x in transitions],bool)
        labels,mask=horizon_labels(costs,self.options['horizon'])
        # A recovery is imitated only after an observed return to nominal with no
        # own intervening collision. Right-censored unfinished recoveries excluded.
        for t,row in enumerate(transitions):
            for i in range(4):
                recovered=False
                if row['safety_rows'][i]['mode']!='NOMINAL' and row['safety_rows'][i]['model_checked']:
                    for future in transitions[t+1:]:
                        if future['collision_cost'][i]:break
                        if future['safety_rows'][i]['mode']=='NOMINAL':recovered=True;break
                self.replay.append((row['safety_features'][i].copy(),row['normalized_executed_acceleration'][i].copy(),
                    labels[t,i].copy(),mask[t,i].copy(),recovered))

    def update(self):
        o=self.options
        if len(self.replay)<o['warmup_samples']:return None
        indices=self.rng.choice(len(self.replay),size=o['batch_size'],replace=False)
        data=[self.replay[int(i)] for i in indices]
        z,u,y,m=[torch.as_tensor(np.asarray([x[j] for x in data]),dtype=torch.float32) for j in range(4)]
        imitate=torch.as_tensor([x[4] for x in data],dtype=torch.bool)
        self.critic_optimizer.zero_grad();x=torch.cat((z,u),dim=1)
        loss=sum((nn.functional.binary_cross_entropy_with_logits(net(x),y,reduction='none')*m).sum()/m.sum().clamp_min(1) for net in self.critics)
        loss.backward();self.critic_optimizer.step()
        for p in self.critics.parameters():p.requires_grad_(False)
        try:
            self.actor_optimizer.zero_grad();proposal=self.actor(z)
            risks=torch.stack([torch.sigmoid(net(torch.cat((z,proposal),1))[:,-1]) for net in self.critics])
            actor_loss=risks.max(0).values.mean()
            if bool(imitate.any()):actor_loss=actor_loss+o['recovery_imitation_weight']*nn.functional.mse_loss(proposal[imitate],u[imitate])
            actor_loss.backward();self.actor_optimizer.step()
        finally:
            for p in self.critics.parameters():p.requires_grad_(True)
        self.updates+=1
        return dict(complete_safety_updates=self.updates,risk_loss=float(loss.detach()),actor_loss=float(actor_loss.detach()),
            observed_horizon_labels=int(m.sum()),successful_recovery_imitation_samples=int(imitate.sum()),
            task_reward_or_optimizer_changed=False)

    def state(self):
        return dict(options=copy.deepcopy(self.options),input_dim=self.input_dim,updates=self.updates,
            actor=copy.deepcopy(self.actor.state_dict()),critics=copy.deepcopy(self.critics.state_dict()))

    def load_for_evaluation(self,state):
        if state['options']!=self.options or state['input_dim']!=self.input_dim:raise ValueError('safety policy contract mismatch')
        if type(state['updates']) is not int or state['updates']<self.options['minimum_updates']:
            raise ValueError('untrained safety policy cannot be used as learned evaluation')
        self.actor.load_state_dict(state['actor'],strict=True);self.critics.load_state_dict(state['critics'],strict=True)
        for p in list(self.actor.parameters())+list(self.critics.parameters()):
            if not torch.isfinite(p).all():raise ValueError('nonfinite safety policy')
        self.updates=state['updates'];self.actor.eval();self.critics.eval()
