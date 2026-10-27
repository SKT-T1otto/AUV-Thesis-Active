"""Goal-conditioned competing-event TD learning and joint goal/residual policy.

Reach and collision are mutually exclusive absorbing FIRST events. A target
actor and the recorded, weight-independent action library define continuation;
this is not a factual-mixture risk classifier. Values remain approximate,
partially observed estimates, never calibrated safety certificates.
"""
from collections import deque
from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch import nn
from chapter3_bser.reach_avoid.options import INPUT_DIM
from chapter3_bser.reach_avoid.goals import Goal, STANDBY, CAPTURE, VIA, encode_goal, goal_reached, next_goal
from chapter3_bser.reach_avoid.control import project
from chapter3_bser.reach_avoid.planning import rule_goal


def first_event_cdf(logits):
    """Competing hazards -> monotone CDFs, with reach+collision <=1 at every h."""
    probabilities = torch.softmax(logits.reshape(*logits.shape[:-1], -1, 3), dim=-1)
    survival = torch.cumprod(probabilities[..., 2], dim=-1)
    before = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), dim=-1)
    reach = torch.cumsum(before * probabilities[..., 0], dim=-1)
    collision = torch.cumsum(before * probabilities[..., 1], dim=-1)
    return torch.stack((reach, collision), dim=-1)


def bellman_events(next_values, *, reached, collision, deadline):
    """Collision wins a simultaneous physical transition; no .95 on probabilities."""
    target = torch.zeros_like(next_values)
    target[:, 1:] = next_values[:, :-1]
    target[deadline] = 0.
    target[reached] = torch.tensor([1., 0.], device=target.device)
    target[collision] = torch.tensor([0., 1.], device=target.device)
    return target


@dataclass
class Sample:
    packet: object
    next_packet: object
    agent: int
    goal: Goal
    action: np.ndarray
    reward: float
    done: bool
    collision: bool
    task_label: bool
    positive_behavior: bool
    intervened: bool


class StratifiedReplay:
    def __init__(self, capacity):
        self.pools = [deque(maxlen=capacity // 3) for _ in range(3)]

    def __len__(self):
        return sum(map(len, self.pools))

    def add(self, sample):
        category = 1 if sample.collision else 0 if sample.positive_behavior else 2
        self.pools[category].append(sample)

    @property
    def counts(self):
        return dict(reach=len(self.pools[0]), collision=len(self.pools[1]), other=len(self.pools[2]))

    def draw(self, count, rng):
        present = [i for i, p in enumerate(self.pools) if p]
        if not present or count < len(present):
            raise ValueError('batch must cover present replay strata')
        quotas = {i:count // len(present) for i in present}
        for i in present[:count % len(present)]:
            quotas[i] += 1
        rows, weights = [], []
        for i in present:
            pool, n = self.pools[i], quotas[i]
            for j in rng.choice(len(pool), n, replace=len(pool) < n):
                rows.append(pool[int(j)])
                # Correct class-balanced sampling back to the retained-buffer
                # distribution; oversampling must not redefine a probability.
                weights.append(count * len(pool) / (len(self) * n))
        return rows, np.asarray(weights, np.float32)


class ReachAvoidLearner:
    def __init__(self, options, *, ablation='full'):
        if ablation not in ('full', 'safety_only', 'no_goal_repair'):
            raise ValueError('unknown reach-avoid ablation')
        self.options = copy.deepcopy(options)
        self.ablation = ablation
        self.rng = np.random.default_rng(options['seed'])
        self.updates = 0
        self.dual = 1.
        self._evaluation_ready = False
        self.intervention = 'none'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options['seed'])
            h, horizon = options['hidden_dim'], options['horizon']
            self.actor = nn.Sequential(nn.Linear(INPUT_DIM,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,3), nn.Tanh())
            self.critics = nn.ModuleList([nn.Sequential(nn.Linear(INPUT_DIM+3,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,3*horizon+1)) for _ in range(2)])
            for net in self.critics:
                with torch.no_grad():
                    net[-1].bias[:-1].reshape(horizon,3)[:] = torch.tensor([-4.,-4.,0.])
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critics = copy.deepcopy(self.critics)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=options['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=options['critic_lr'])
        self.replay = StratifiedReplay(options['replay_size'])

    @property
    def ready(self):
        c = self.replay.counts
        return self._evaluation_ready or (self.updates >= self.options['minimum_updates']
            and len(self.replay) >= self.options['warmup_samples'] and c['reach'] > 0 and c['collision'] > 0)

    @staticmethod
    def inputs(packet, agent, goal):
        return np.concatenate((packet.features[agent], encode_goal(packet, agent, goal))).astype(np.float32)

    def residual(self, packet, agent, goal, *, explore=False, target=False):
        if self.intervention=='zero_residual':return np.zeros(3,np.float32)
        network = self.target_actor if target else self.actor
        with torch.no_grad():
            action = network(torch.as_tensor(self.inputs(packet, agent, goal))).numpy().copy()
        if explore:
            action = np.clip(action + self.rng.normal(0., .15, 3), -1., 1.)
        return action

    def policy_command(self, packet, agent, goal, *, target=False):
        x = self.inputs(packet, agent, goal)
        command = np.clip(x[-3:] + self.residual(packet, agent, goal, target=target), -1., 1.)
        if goal.kind in (STANDBY,CAPTURE) and goal_reached(packet,agent,goal,self.options):command=x[-3:]
        result, index = project(command, packet.actions[agent], packet.action_mask[agent])
        if packet.action_mask[agent,9] and packet.checked[agent,9]:
            return packet.actions[agent,9].copy()
        if ((self.ablation == 'no_goal_repair' or self.intervention=='no_goal_repair') and not packet.goal_mask[agent,0]
            and np.allclose(goal.array(), packet.goals[agent,0], rtol=0, atol=1e-6)):
            result, index = project(packet.actions[agent,8], packet.actions[agent], packet.action_mask[agent])
        return result

    def values(self, packet, agent, goal, *, target=False):
        values, _ = self.evaluate_queries([(packet,agent,goal)],target=target)
        return values[0]

    def evaluate_queries(self, queries, *, target=False):
        """Batch both networks; projection is the same fixed-library policy."""
        x = np.array([self.inputs(p,i,g) for p,i,g in queries])
        actor = self.target_actor if target else self.actor
        nets = self.target_critics if target else self.critics
        with torch.no_grad():
            residuals=np.zeros((len(x),3),np.float32) if self.intervention=='zero_residual' else actor(torch.tensor(x)).numpy()
            proposals = np.clip(x[:,-3:]+residuals,-1.,1.)
            commands=[]
            for (p,i,g), u in zip(queries,proposals):
                if g.kind in (STANDBY,CAPTURE) and goal_reached(p,i,g,self.options):u=self.inputs(p,i,g)[-3:]
                a,_=project(u,p.actions[i],p.action_mask[i])
                if p.action_mask[i,9] and p.checked[i,9]:a=p.actions[i,9]
                if (self.ablation=='no_goal_repair' or self.intervention=='no_goal_repair') and not p.goal_mask[i,0] and not (p.action_mask[i,9] and p.checked[i,9]) and np.allclose(g.array(),p.goals[i,0],rtol=0,atol=1e-6):
                    a,_=project(p.actions[i,8],p.actions[i],p.action_mask[i])
                commands.append(a)
            outputs=torch.stack([net(torch.tensor(np.concatenate((x,np.array(commands)),1))) for net in nets])
            events=first_event_cdf(outputs[:,:,:-1])
            combined=torch.stack((events[:,:,:,0].min(0).values,events[:,:,:,1].max(0).values),-1)
            task=outputs[:,:,-1].min(0).values
        return [dict(reach=float(e[-1,0]),risk=float(e[-1,1]),task=float(t),calibrated_probability_bound=False)
                for e,t in zip(combined,task)],combined

    def choose_scores(self, packet, agent, scores, ablation, explore=False):
        valid=[int(j) for j in np.flatnonzero(packet.goal_mask[agent])]
        if ablation=='no_goal_repair':return 0,'learned_original_goal_only'
        if packet.phases[agent]==2:
            ready=[j for j in valid if Goal.from_array(packet.goals[agent,j]).kind==CAPTURE
                   and goal_reached(packet,agent,Goal.from_array(packet.goals[agent,j]),self.options)]
            if ready:return ready[0],'capture_region_hold_original_completion_protocol'
        if explore and valid and self.rng.random()<.15:return int(self.rng.choice(valid)),'goal_exploration'
        eligible=[j for j in valid if scores[j]['risk']<=self.options['risk_limit']
                  and (ablation=='safety_only' or scores[j]['reach']>=self.options['reach_floor'])]
        if not eligible:return rule_goal(packet,agent),'value_constraints_empty_rule_fallback'
        def score(j):
            v=scores[j]
            return v['task']-self.dual*v['risk']+(0. if ablation=='safety_only' else self.options['reach_weight']*v['reach'])
        return max(eligible,key=score),'learned_goal_and_residual'

    def select_goal(self, packet, agent, *, ablation=None, explore=False, target=False):
        ablation = self.ablation if ablation is None else ablation
        if self.intervention=='no_goal_repair':ablation='no_goal_repair'
        scores,_=self.evaluate_queries([(packet,agent,Goal.from_array(g)) for g in packet.goals[agent]],target=target)
        index,reason=self.choose_scores(packet,agent,scores,ablation,explore)
        return index,scores,reason

    def add_episode(self, records):
        if not records:
            raise ValueError('empty reach-avoid episode')
        for t, row in enumerate(records):
            for i in range(4):
                if row['packet'].phases[i] == 3:
                    continue
                actual = Goal.from_array(row['goals'][i])
                goals = [(actual, True)]
                original = Goal.from_array(row['packet'].goals[i,0])
                if not np.allclose(original.array(), actual.array(), rtol=0, atol=1e-6):
                    goals.append((original, False))
                # Hindsight changes only the auxiliary geometric event. Original
                # task rewards are NEVER copied onto an unexecuted goal choice.
                future_indices = list(range(t, min(len(records), t+self.options['horizon'])))
                self.rng.shuffle(future_indices)
                added = 0
                for j in future_indices:
                    future = records[j]
                    if any(x['team_collision'] for x in records[t:j+1]) or future['next_packet'].phases[i] != row['packet'].phases[i]:
                        continue
                    goal = Goal(tuple(future['next_packet'].positions[i]), kind=STANDBY if row['packet'].phases[i] == 1 else VIA)
                    if goal_reached(future['next_packet'], i, goal, self.options) and not goal_reached(row['packet'], i, goal, self.options):
                        goals.append((goal, False));added += 1
                    if added >= self.options['hindsight_per_agent']:
                        break
                for goal, factual in goals:
                    if goal_reached(row['packet'], i, goal, self.options):
                        continue
                    positive = False
                    current_goal = goal
                    for future in records[t:t+self.options['horizon']]:
                        if future['team_collision']:
                            break
                        current_goal = next_goal(current_goal, future['next_packet'], i)
                        if goal_reached(future['next_packet'], i, current_goal, self.options):
                            positive = True;break
                        if future['next_packet'].phases[i] != row['packet'].phases[i]:
                            break
                    self.replay.add(Sample(row['packet'], row['next_packet'], i, goal,
                        row['actual_actions'][i].copy(), float(row['reward']), bool(row['done']),
                        bool(row['team_collision']), factual, positive, bool(row['rows'][i]['projection_changed'])))

    def update(self):
        if self.intervention!='none':raise ValueError('evaluation interventions cannot train or relabel policies')
        o = self.options
        if len(self.replay) < o['warmup_samples']:
            return None
        batch, weights = self.replay.draw(o['batch_size'], self.rng)
        x = torch.tensor(np.array([self.inputs(s.packet,s.agent,s.goal) for s in batch]))
        a = torch.tensor(np.array([s.action for s in batch]))
        w = torch.tensor(weights)
        factual = torch.tensor([s.task_label for s in batch], dtype=torch.bool)
        task_targets, reached, failures, deadlines = [], [], [], []
        with torch.no_grad():
            goals=[next_goal(s.goal,s.next_packet,s.agent) for s in batch]
            _,next_values=self.evaluate_queries([(s.next_packet,s.agent,g) for s,g in zip(batch,goals)],target=True)
            factual_next=[j for j,s in enumerate(batch) if s.task_label and not s.done]
            queries=[(batch[j].next_packet,batch[j].agent,Goal.from_array(g)) for j in factual_next for g in batch[j].next_packet.goals[batch[j].agent]]
            next_tasks={}
            if queries:
                scores,_=self.evaluate_queries(queries,target=True)
                for offset,j in enumerate(factual_next):
                    s=batch[j];values=scores[offset*o['candidates']:(offset+1)*o['candidates']]
                    index,_=self.choose_scores(s.next_packet,s.agent,values,self.ablation)
                    next_tasks[j]=values[index]['task']
            for j,(s,g) in enumerate(zip(batch,goals)):
                success = goal_reached(s.next_packet, s.agent, g, o)
                deadline = s.done or s.next_packet.phases[s.agent] != s.packet.phases[s.agent]
                reached.append(success);failures.append(s.collision);deadlines.append(deadline)
                if s.done:
                    task_targets.append(s.reward)
                else:
                    value = next_tasks.get(j,0.)
                    task_targets.append(s.reward + o['gamma'] * value)
            target = bellman_events(next_values, reached=torch.tensor(reached),
                collision=torch.tensor(failures), deadline=torch.tensor(deadlines))
            tq = torch.tensor(task_targets, dtype=torch.float32)
        self.critic_optimizer.zero_grad()
        critic_loss = 0.
        for net in self.critics:
            output = net(torch.cat((x,a),1))
            event = first_event_cdf(output[:,:-1]).clamp(1e-6,1-1e-6)
            error = nn.functional.binary_cross_entropy(event,target,reduction='none').mean((1,2))
            critic_loss = critic_loss + (error*w).sum()/w.sum()
            if bool(factual.any()):
                critic_loss = critic_loss + (((output[:,-1]-tq)**2)*w*factual).sum()/(w*factual).sum()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(),10.)
        self.critic_optimizer.step()
        for p in self.critics.parameters():p.requires_grad_(False)
        try:
            self.actor_optimizer.zero_grad()
            proposal = (x[:,-3:] + self.actor(x)).clamp(-1,1)
            projected = []
            for s, u in zip(batch,proposal.detach().numpy()):
                command, _ = project(u,s.packet.actions[s.agent],s.packet.action_mask[s.agent])
                if s.packet.action_mask[s.agent,9] and s.packet.checked[s.agent,9]:command=s.packet.actions[s.agent,9]
                if self.ablation=='no_goal_repair' and not s.packet.goal_mask[s.agent,0] and not (s.packet.action_mask[s.agent,9] and s.packet.checked[s.agent,9]) and np.allclose(s.goal.array(),s.packet.goals[s.agent,0],rtol=0,atol=1e-6):
                    command,_=project(s.packet.actions[s.agent,8],s.packet.actions[s.agent],s.packet.action_mask[s.agent])
                projected.append(command)
            projected = torch.tensor(np.array(projected))
            # Explicit straight-through surrogate for the nondifferentiable
            # projection. Forward values use executable commands, not fantasies.
            executed = proposal + (projected-proposal).detach()
            outputs = torch.stack([net(torch.cat((x,executed),1)) for net in self.critics])
            cdf = first_event_cdf(outputs[:,:,:-1])
            reach = cdf[:,:,-1,0].min(0).values
            risk = cdf[:,:,-1,1].max(0).values
            task = outputs[:,:,-1].min(0).values
            actor_loss = self.dual * risk.mean()
            if self.ablation != 'safety_only':actor_loss = actor_loss-o['reach_weight']*reach.mean()
            if bool(factual.any()):actor_loss = actor_loss-task[factual].mean()
            actor_loss = actor_loss+o['projection_weight']*((proposal-projected)**2).mean()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),10.)
            self.actor_optimizer.step()
        finally:
            for p in self.critics.parameters():p.requires_grad_(True)
        self.dual=float(np.clip(self.dual+o['dual_lr']*(float(risk.mean().detach())-o['risk_limit']),0,o['dual_max']))
        for target_net, net in ((self.target_actor,self.actor),(self.target_critics,self.critics)):
            with torch.no_grad():
                for target_p, p in zip(target_net.parameters(),net.parameters()):target_p.lerp_(p,o['target_tau'])
        self.updates+=1
        return dict(complete_updates=self.updates,event_loss_and_task_loss=float(critic_loss.detach()),actor_loss=float(actor_loss.detach()),
            dual=self.dual,retained_strata=self.replay.counts,task_reward_changed=False,policy_evaluation='goal-conditioned projected residual actor',
            actor_gradient='straight-through projection surrogate',calibrated_probability_bound=False)

    def state(self):
        return dict(options=copy.deepcopy(self.options),input_dim=INPUT_DIM,ablation=self.ablation,updates=self.updates,dual=self.dual,
            coverage=self.replay.counts,actor=copy.deepcopy(self.actor.state_dict()),critics=copy.deepcopy(self.critics.state_dict()))

    def load_for_evaluation(self,state):
        if state.get('options')!=self.options or state.get('input_dim')!=INPUT_DIM or state.get('ablation')!=self.ablation:
            raise ValueError('independent reach-avoid policy identity mismatch')
        if state.get('updates',0)<self.options['minimum_updates'] or min(state.get('coverage',{}).get(k,0) for k in ('reach','collision'))<1:
            raise ValueError('untrained or unsupported reach-avoid policy')
        self.actor.load_state_dict(state['actor'],strict=True);self.critics.load_state_dict(state['critics'],strict=True)
        if any(not torch.isfinite(p).all() for p in list(self.actor.parameters())+list(self.critics.parameters())):
            raise ValueError('nonfinite reach-avoid policy')
        self.updates=state['updates'];self.dual=float(state['dual']);self._evaluation_ready=True
        self.actor.eval();self.critics.eval()
