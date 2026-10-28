"""Replayable action projection, shared by rule and learned comparisons.

The action library depends only on public state, not on network weights. TD
targets therefore use exactly the same projection as deployed residual actors.
"""
import numpy as np
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.options import validate as safety_options
from chapter3_bser.safe_rl.dynamics import brake, limits, integrate
from chapter3_bser.reach_avoid.scan_geometry import RaySubcells, ScanRefinedGeometry, refined_buffer_exit
from chapter3_bser.reach_avoid.options import SPECS, CONTRACT
from chapter3_bser.reach_avoid.features import Encoder
from chapter3_bser.reach_avoid.goals import Goal, CAPTURE, STANDBY, base_action, encode_goal, goal_reached
from chapter3_bser.reach_avoid.planning import candidate_goals, rule_goal


def project(command, actions, mask):
    valid = np.flatnonzero(mask)
    if not len(valid):
        raise ValueError('action library must expose its explicitly unverified emergency fallback')
    index = int(valid[np.argmin(np.sum((actions[valid] - command) ** 2, axis=1))])
    return actions[index].copy(), index


class ReachAvoidController:
    def __init__(self, specs, contract, options, max_steps, learner=None, ablation='full'):
        for declared, actual in zip(SPECS, specs):
            if any(declared[k] != actual[k] for k in declared):
                raise ValueError('public vehicle contract changed')
        mismatch={k:(v,contract.get(k)) for k,v in CONTRACT.items() if contract.get(k)!=v}
        if mismatch:
            raise ValueError('public control contract changed: '+str(mismatch))
        self.options = options
        self.guard = SafetyController(specs, contract, safety_options())
        self.subcells = RaySubcells(contract['bounds'])
        self.encoder = Encoder(options, max_steps)
        self.learner = learner
        self.ablation = ablation
        self.packet = self.geometry = None
        self.trace = []
        self.bank_labels = None
        self.rng = np.random.default_rng(options['seed'])

    def snapshot(self, state, task, observations, semantic, delivered=None):
        if self.packet is not None and self.packet.step == state.step:
            return self.packet
        self.guard.observe(state, self._events)
        self.subcells.consume(self._events, state.step)
        geometry = ScanRefinedGeometry(state, self.guard.evidence, self.guard.options, self.subcells)
        packet = self.encoder.build(state, task, observations, semantic, geometry, self.guard,delivered)
        packet.goals, packet.goal_mask = candidate_goals(packet, state, semantic, geometry, self.options)
        banks, masks, checked, labels = [], [], [], []
        for i, spec in enumerate(SPECS):
            p, v = packet.positions[i], packet.velocities[i]
            candidates = []
            for j, raw in enumerate(packet.goals[i]):
                candidates.append((f'GOAL_BASE_{j}', base_action(packet, i, Goal.from_array(raw)), bool(packet.goal_mask[i, j])))
            b = brake(v, spec, CONTRACT['dt'], packet.flow[i]) / limits(spec)
            candidates.append(('BRAKE', b, True))
            recovery = refined_buffer_exit(p, v, spec, geometry, self.guard.contract, self.guard.options, packet.flow[i])
            candidates.append(('BUFFER_EXIT', b if recovery is None else recovery[0] / limits(spec), recovery is not None))
            for j in range(3):
                for sign in (-1, 1):
                    direction = np.eye(3)[j] * .5 * sign
                    candidates.append((f'AXIS_{j}_{sign}', np.clip(b + direction, -1., 1.), True))
            bank, valid, certified, names = [], [], [], []
            caps = self.guard._caps(p, v, spec, geometry)
            violation = max(0., np.linalg.norm(v[:2]) - caps[0]) + max(0., abs(v[2]) - caps[1])
            for label, action, eligible in candidates:
                action = np.asarray(action, float)
                ok = False
                if eligible:
                    if label == 'BUFFER_EXIT':
                        ok = True
                    else:
                        nxt_v = integrate(p, v, action * limits(spec), spec, CONTRACT['dt'], packet.flow[i])[1]
                        next_violation = max(0., np.linalg.norm(nxt_v[:2]) - caps[0]) + max(0., abs(nxt_v[2]) - caps[1])
                        speed_ok = next_violation <= 1e-6 or next_violation < violation - 1e-8
                        if speed_ok:
                            ok = self.guard.check(p, v, action * limits(spec), spec, geometry, packet.flow[i])[0]
                bank.append(action);valid.append(ok);certified.append(ok);names.append(label)
            if not any(valid):
                valid[8] = True
                names[8] = 'UNCERTIFIED_BRAKE'
            banks.append(bank);masks.append(valid);checked.append(certified);labels.append(names)
        packet.actions = np.asarray(banks, np.float32)
        packet.action_mask = np.asarray(masks, bool)
        packet.checked = np.asarray(checked, bool)
        self.packet, self.geometry, self.bank_labels = packet, geometry, labels
        return packet

    def observe(self, state, task, observations, semantic, events, delivered=None):
        self._events = events
        return self.snapshot(state, task, observations, semantic,delivered)

    def act(self, packet, explore=False):
        outputs, goals, rows = [], [], []
        for i in range(4):
            selected = rule_goal(packet, i)
            reason = 'shared_rule_baseline'
            scores = None
            if self.learner is not None and self.learner.ready and packet.phases[i] != 3:
                selected, scores, reason = self.learner.select_goal(packet, i, ablation=self.ablation, explore=explore)
            elif explore and packet.phases[i] != 3:
                valid=np.flatnonzero(packet.goal_mask[i])
                if len(valid) and self.rng.random()<.3:
                    selected=int(self.rng.choice(valid));reason='untrained_goal_exploration'
            if self.ablation == 'no_goal_repair':
                selected = 0
                reason += ':original_goal_only'
            elif packet.phases[i]==2:
                ready=[j for j,g in enumerate(packet.goals[i]) if packet.goal_mask[i,j]
                       and Goal.from_array(g).kind==CAPTURE and goal_reached(packet,i,Goal.from_array(g),self.options)]
                if ready:selected=ready[0];reason='capture_region_hold_original_completion_protocol'
            goal = Goal.from_array(packet.goals[i, selected])
            base = base_action(packet, i, goal)
            proposal = base.copy()
            terminal_hold=goal.kind in (STANDBY,CAPTURE) and goal_reached(packet,i,goal,self.options)
            if self.learner is not None and self.learner.ready and packet.phases[i] != 3 and not terminal_hold:
                residual = self.learner.residual(packet, i, goal, explore=explore)
                proposal = np.clip(base + residual, -1., 1.)
            command, action_index = project(proposal, packet.actions[i], packet.action_mask[i])
            if explore and (self.learner is None or not self.learner.ready) and packet.phases[i]!=3 and self.rng.random()<.2:
                action_index=int(self.rng.choice(np.flatnonzero(packet.action_mask[i])))
                command=packet.actions[i,action_index].copy()
            # Verified margin exit remains available even if no goal is currently
            # executable. It is not vetoed by an uncalibrated probability head.
            if packet.action_mask[i, 9] and self.bank_labels[i][9] == 'BUFFER_EXIT':
                command, action_index = packet.actions[i, 9].copy(), 9
            if self.ablation == 'no_goal_repair' and not packet.goal_mask[i, 0] and not packet.action_mask[i, 9]:
                command, action_index = project(packet.actions[i, 8], packet.actions[i], packet.action_mask[i])
            outputs.append(command);goals.append(goal)
            rows.append(dict(step=packet.step, agent_id=i, phase=int(packet.phases[i]),
                raw_assigned_goal=packet.goals[i, 0].tolist(), selected_goal=goal.array().tolist(),
                selected_index=selected, original_goal_admissible=bool(packet.goal_mask[i, 0]),
                selection_reason=reason, candidate_values=scores, action_index=action_index,
                action_mode=self.bank_labels[i][action_index], model_checked=bool(packet.checked[i, action_index]),
                public_position=packet.positions[i].tolist(), velocity=packet.velocities[i].tolist(),
                baseline_action=base.tolist(), proposed_action=proposal.tolist(), executed_action=command.tolist(),
                projection_changed=bool(np.linalg.norm(command-proposal) > 1e-5),
                local_geometry=self.geometry.diagnostics(packet.positions[i]),
                stop_cause='no_public_model_checked_action' if self.bank_labels[i][action_index]=='UNCERTIFIED_BRAKE' else
                    'checked_braking' if self.bank_labels[i][action_index]=='BRAKE' else None,
                learned_probability_is_certified=False, safety_guarantee=False))
        self.guard.previous_v = packet.velocities.copy()
        self.guard.previous = np.asarray(outputs) * np.array([limits(s) for s in SPECS])
        self.trace.extend(rows)
        return np.asarray(outputs, np.float32), goals, rows
