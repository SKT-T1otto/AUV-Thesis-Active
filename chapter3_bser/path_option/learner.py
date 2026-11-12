"""Off-policy masked actor-critic with one-step option continuation targets.

Task returns use gamma=.95 once per physical step. Event probabilities use an
undiscounted 32-step competing-event recursion. Neither is a safety certificate.
"""
import copy
from collections import deque
import numpy as np
import torch
from torch import nn
from chapter3_bser.reach_avoid.learner import first_event_cdf, bellman_events
from chapter3_bser.path_option.networks import Policy, Critic, tensor_batch, masked_distribution
from chapter3_bser.path_option.options import validate, VARIANTS


def option_return(reward, done, continued, switched, beta, gamma=.95):
    return reward + gamma*(1-done)*((1-beta)*continued+beta*switched)


class Learner:
    def __init__(self, options=None, variant='A'):
        self.o = validate(options)
        if variant not in VARIANTS:
            raise ValueError('unknown trained variant')
        self.variant = variant
        self.rng = np.random.default_rng(self.o['seed'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.o['seed'])
            self.policy = Policy(self.o, variant)
            self.critics = nn.ModuleList([Critic(self.o, variant) for _ in range(2)])
        self.target_policy = copy.deepcopy(self.policy)
        self.target_critics = copy.deepcopy(self.critics)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.o['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.o['critic_lr'])
        self.replay = deque(maxlen=self.o['replay_size'])
        self.updates = 0
        self.evaluation_only = False

    def distributions(self, batch, rows, *, target=False, _critic_encodings=None):
        nets = self.target_critics if target else self.critics
        policy = self.target_policy if target else self.policy
        with torch.no_grad():
            if _critic_encodings is None:
                outputs = [c(batch) for c in nets]
            else:
                outputs = [c.forward_encoded(batch, z) for c, z in zip(nets, _critic_encodings)]
            qs, hazards = zip(*outputs)
        q = torch.stack(qs).min(0).values
        cdf = torch.stack([first_event_cdf(h) for h in hazards])
        reach = cdf[..., -1, 0].min(0).values
        risk = cdf[..., -1, 1].max(0).values
        # Goal scores also backpropagate through event expectations. Preserve
        # the two original actor graphs/accumulation order when gradients are
        # enabled: merging them perturbs near-zero gradients amplified by Adam.
        # Inference/targets have no graph and safely reuse one policy encoding.
        if torch.is_grad_enabled():
            low0, _, _ = policy(batch, torch.zeros((*q.shape[:2], 2)))
        else:
            z = policy.encoder(batch)
            low, high = policy.action_goal(batch, z)
            low0 = low
        # The action distribution used for values includes deployed forced overrides.
        force = torch.tensor([r['forced'] for r in rows], dtype=torch.long)
        forced = force >= 0
        hot = nn.functional.one_hot(force.clamp_min(0), q.shape[-1]).float()
        low0 = torch.where(forced[:, None, None], hot[:, None], low0)
        events = torch.stack(((low0*reach).sum(-1), (low0*risk).sum(-1)), -1)
        if self.variant == 'D':
            events = torch.zeros_like(events)
        elif self.variant == 'risk_only':
            events = torch.stack((events[..., 0]*0, events[..., 1]), -1)
        if torch.is_grad_enabled():
            low, high, beta = policy(batch, events)
        else:
            beta = policy.termination_probability(z, events)
        low = torch.where(forced[:, None, None], hot[:, None], low)
        high = high + self.o['reach_weight']*events[..., 0] - self.o['risk_weight']*events[..., 1]
        mu = masked_distribution(high, batch['goal_mask'])
        for i, r in enumerate(rows):
            if self.variant == 'rule_recovery' or r.get('intervention') == 'rule_recovery':
                mu = mu.clone(); mu[i] = 0.; mu[i, r['rule_goal']] = 1.
            mode = 'periodic' if self.variant == 'periodic' else 'rule' if self.variant == 'C' or r.get('intervention') == 'rule_termination' else 'learned'
            value = r['periodic_terminate'] if mode == 'periodic' else r['rule_terminate'] if mode == 'rule' else None
            if not r['beta_allowed']:
                value = r['forced_terminate']
            if value is not None:
                beta = beta.clone(); beta[i, 0] = float(value)
        return low, mu, beta, q, reach, risk, events

    @torch.no_grad()
    def decide(self, row, *, explore=False):
        b = tensor_batch([row], row.get('intervention', 'none'))
        low, mu, beta, q, reach, risk, events = self.distributions(b, [row])
        # Stochastic option termination is the trained beta policy, also during evaluation.
        terminate = bool(self.rng.random() < float(beta[0, 0]))
        j = int(self.rng.choice(8, p=mu[0].numpy()/float(mu[0].sum()))) if terminate else 0
        probs = low[0, j].numpy()
        # Evaluation deploys the SAME categorical policy evaluated by the critic.
        # Switching to argmax here would silently change the reach/return policy.
        k = int(self.rng.choice(len(probs), p=probs/probs.sum()))
        return dict(goal=j, action=k, terminate=terminate, beta=float(beta[0, 0]),
            reach=float(events[0, j, 0]), risk=float(events[0, j, 1]),
            task=float((low*q)[0, j].sum()), action_probabilities=probs.tolist())

    def add(self, transition):
        if self.evaluation_only:
            raise ValueError('evaluation policy cannot learn')
        # Factual task labels only. No rewards are assigned to unexecuted goals.
        self.replay.append(transition)

    def update(self):
        if self.evaluation_only:
            raise ValueError('checkpoint restore is evaluation-only')
        if len(self.replay) < self.o['warmup']:
            return None
        ids = self.rng.choice(len(self.replay), self.o['batch_size'], replace=False)
        rows = [self.replay[int(i)] for i in ids]
        states = [r['state'] for r in rows]; nexts = [r['next'] for r in rows]
        b = tensor_batch(states); nb = tensor_batch(nexts)
        j = torch.tensor([r['goal_index'] for r in rows]); ix = torch.arange(len(rows))
        actual = torch.tensor(np.array([r['actual_action'] for r in rows]))[:, None]
        rewards = torch.tensor([r['reward'] for r in rows], dtype=torch.float32)
        done = torch.tensor([r['done'] for r in rows], dtype=torch.float32)
        with torch.no_grad():
            # The second target query changes only the event region. Reuse
            # encodings before any optimizer step, including all route tokens.
            target_encodings = [c.encoder(nb) for c in self.target_critics]
            lp, mu, beta, nq, _, _, _ = self.distributions(
                nb, nexts, target=True, _critic_encodings=target_encodings)
            qo = (lp*nq).sum(-1)
            target_q = option_return(rewards, done, qo[:, 0], (mu*qo).sum(-1), beta[:, 0], self.o['gamma'])
            # Keep the ORIGINAL event region fixed across the transition.
            query = np.array([(r['event_target']-r['next']['position'])/r['next']['bounds'] for r in rows])
            query = torch.tensor(query, dtype=torch.float32)[:, None].expand(-1, 8, -1)
            out = [c.forward_encoded(nb, z, event_relative=query)[1]
                   for c, z in zip(self.target_critics, target_encodings)]
            events = torch.stack([first_event_cdf(x)[:, 0] for x in out])
            # Avoid mixing min/max hazards across individual actions before expectation.
            pred = torch.stack(((events[..., 0]*lp[None, :, 0, :, None]).sum(2).min(0).values,
                                (events[..., 1]*lp[None, :, 0, :, None]).sum(2).max(0).values), -1)
            target_event = bellman_events(pred,
                reached=torch.tensor([r['reached'] for r in rows]),
                collision=torch.tensor([r['collision'] for r in rows]),
                deadline=torch.tensor([r['event_deadline'] for r in rows]))
        self.critic_optimizer.zero_grad()
        critic_loss = 0.
        event_valid = torch.tensor([s.get('phase', 0) != 3 for s in states])
        for c in self.critics:
            q, h = c(b, actual)
            critic_loss = critic_loss+nn.functional.mse_loss(q[ix, j, 0], target_q)
            if self.variant != 'D' and bool(event_valid.any()):
                event = first_event_cdf(h[ix, j, 0])[event_valid].clamp(1e-6, 1-1e-6)
                if self.variant == 'risk_only':
                    critic_loss = critic_loss+nn.functional.binary_cross_entropy(event[..., 1], target_event[event_valid, :, 1])
                else:
                    critic_loss = critic_loss+nn.functional.binary_cross_entropy(event, target_event[event_valid])
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), 10.)
        self.critic_optimizer.step()
        # Enumerated categorical actions: no nearest-neighbour straight-through gradient.
        self.actor_optimizer.zero_grad()
        # Recompute after the critic step, then reuse its detached values for
        # actor utility instead of repeating the identical distribution query.
        lp, mu, beta, q, reach, risk, _ = self.distributions(b, states)
        utility = q.detach()
        if self.variant != 'D':
            utility = utility-self.o['risk_weight']*risk.detach()
            if self.variant != 'risk_only':
                utility = utility+self.o['reach_weight']*reach.detach()
        low_value = (lp*utility).sum(-1)
        active = torch.tensor([r['policy_executed'] for r in rows], dtype=torch.float32)
        low_loss = -(low_value[ix, j]*active).sum()/active.sum().clamp_min(1.)
        task_option = (lp.detach()*q.detach()).sum(-1)
        decisions = torch.tensor([r['decision'] for r in rows], dtype=torch.float32)
        high_loss = -((mu*low_value.detach()).sum(-1)*decisions).sum()/decisions.sum().clamp_min(1.)
        allowed = torch.tensor([s['beta_allowed'] for s in states], dtype=torch.float32)
        if self.variant in ('C', 'periodic'):
            allowed.zero_()
        advantage = task_option[:, 0]-(mu.detach()*task_option).sum(-1)
        beta_loss = (beta[:, 0]*advantage.detach()*allowed).sum()/allowed.sum().clamp_min(1.)
        entropy = -(lp.clamp_min(1e-8).log()*lp).sum(-1)[ix, j]
        loss = low_loss+high_loss+beta_loss-self.o['entropy_weight']*(entropy*active).sum()/active.sum().clamp_min(1.)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.)
        self.actor_optimizer.step()
        for target, current in ((self.target_policy, self.policy), (self.target_critics, self.critics)):
            with torch.no_grad():
                for p, c in zip(target.parameters(), current.parameters()):
                    p.lerp_(c, self.o['tau'])
        self.updates += 1
        return dict(updates=self.updates, critic_loss=float(critic_loss.detach()), actor_loss=float(loss.detach()),
            termination_loss=float(beta_loss.detach()), factual_labels_only=True, straight_through=False)

    def weights(self):
        return dict(policy=self.policy.state_dict(), critics=self.critics.state_dict(), updates=self.updates)

    def load_evaluation(self, weights):
        self.policy.load_state_dict(weights['policy'], strict=True)
        self.critics.load_state_dict(weights['critics'], strict=True)
        if any(not torch.isfinite(p).all() for p in list(self.policy.parameters())+list(self.critics.parameters())):
            raise ValueError('nonfinite weights')
        self.updates = int(weights['updates'])
        self.evaluation_only = True
        self.policy.eval(); self.critics.eval()
