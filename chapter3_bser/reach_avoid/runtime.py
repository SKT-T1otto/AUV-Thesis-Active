"""Independent OLD runtime; original environment/reward/task networks are intact."""
from dataclasses import replace
from collections import Counter
import numpy as np
import torch
from core.mapping.planning_state import extract_planning_state
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.safe_rl.runtime import SafetyRuntime
from chapter3_bser.safe_rl.dynamics import nominal, limits
from chapter3_bser.reach_avoid.options import validate, SCHEMA, INPUT_DIM, SPECS
from chapter3_bser.reach_avoid.goals import Goal, goal_reached, next_goal
from chapter3_bser.reach_avoid.control import ReachAvoidController


class ReachAvoidRuntime(SafetyRuntime):
    def __init__(self, config, scenario, *, seed, episode_id=0, options=None, learner=None,
                 behavior='rule', ablation='full', goal_faults=None):
        if behavior not in ('rule','learned','explore','nominal'):
            raise ValueError('unknown reach-avoid behavior')
        # Reuse only the already reviewed absolute-acceleration input factory.
        # Disable the parent's navigation repair/learner, retaining its original
        # raw D2 bridge and all task/reward/termination implementations.
        super().__init__(config,scenario,seed=seed,episode_id=episode_id,collection_only=True)
        self.ra_options=validate(options)
        self.behavior=behavior
        self.goal_faults={} if goal_faults is None else dict(goal_faults)
        self.ra=ReachAvoidController(self.env.get_public_vehicle_specs(),self.nominal_contract,
            self.ra_options,int(config['max_steps']),learner,ablation)
        self.ra_records=[]
        self.ra_counts=Counter()
        self._last_ra=None
        self._active_exploration=False

    def _semantic(self):
        return {x.agent_id:tuple(self.goal_faults.get(x.agent_id,x.final_waypoint)) for x in self.guidance.agent_assignments}

    def _packet(self):
        state=extract_planning_state(self.env)
        return self.ra.observe(state,self.env.get_task_state(),self.observations,self._semantic(),
            self.env.read_public_events(self.ra.guard.evidence.cursor),self.env.get_public_delivered_target())

    def _actions_for_step(self, actions):
        # Preserve original task-action bookkeeping in EVERY new behavior. The
        # auxiliary recovery command is separate from that task residual gate.
        task=RepairedMissionRuntime._actions_for_step(self,actions).numpy()
        packet=self._packet()
        if self.behavior=='nominal':
            by_id={x.agent_id:x for x in self.guidance.agent_assignments}
            commands=[];goals=[];rows=[]
            for i,spec in enumerate(SPECS):
                item=by_id[i]
                target=item.hold_position if item.hold_state else item.tracking_waypoint
                # Optional explicit fault input is part of the experiment, not
                # a query of obstacle truth or an environment-goal mutation.
                if i in self.goal_faults:target=self.goal_faults[i]
                u,_,_=nominal(packet.positions[i],packet.velocities[i],target,task[i],spec,self.nominal_contract,i)
                command=(u/limits(spec)).astype(np.float32)
                goal=Goal.from_array(packet.goals[i,0])
                commands.append(command);goals.append(goal)
                rows.append(dict(step=packet.step,agent_id=i,phase=int(packet.phases[i]),
                    raw_assigned_goal=packet.goals[i,0].tolist(),selected_goal=goal.array().tolist(),selected_index=0,
                    selection_reason='ORIGINAL_D2_NOMINAL_COLLECTION',action_mode='NOMINAL_COLLECTION',
                    model_checked=False,proposed_action=command.tolist(),executed_action=command.tolist(),
                    projection_changed=False,public_position=packet.positions[i].tolist(),velocity=packet.velocities[i].tolist(),
                    learned_probability_is_certified=False,safety_guarantee=False))
            commands=np.asarray(commands,np.float32)
            self.ra.guard.previous_v=packet.velocities.copy()
        else:
            commands,goals,rows=self.ra.act(packet,explore=self._active_exploration or self.behavior=='explore')
            assignments=[]
            for item,goal in zip(self.guidance.agent_assignments,goals):
                i=item.agent_id
                assignments.append(replace(item,final_waypoint=tuple(goal.center),
                    planned_path=(tuple(packet.positions[i]),tuple(goal.ref)),tracking_waypoint=tuple(goal.ref),
                    hold_position=tuple(goal.ref),hold_state=bool(packet.phases[i]==3),reachable=bool(packet.goal_mask[i,rows[i]['selected_index']])))
            executor=assignments[3]
            self.guidance=replace(self.guidance,agent_assignments=tuple(assignments),
                executor_assignment=replace(self.guidance.executor_assignment,target_region=executor.final_waypoint,
                    planned_path=executor.planned_path,tracking_waypoint=executor.tracking_waypoint,
                    hold_position=executor.hold_position,hold_state=executor.hold_state,reachable=executor.reachable))
            self.env.install_guidance(self.guidance)
        self._last_ra=dict(packet=packet,goals=np.array([g.array() for g in goals]),rows=rows)
        return torch.tensor(commands,dtype=torch.float32)

    def advance(self, model, *, explore=False):
        self._active_exploration=bool(explore)
        row=RepairedMissionRuntime.advance(self,model,explore=False)
        data=self._last_ra
        actual=self.env.unwrapped._agent_acc.detach().cpu().numpy().copy()
        scales=np.array([limits(s) for s in SPECS])
        normalized=(actual/scales).astype(np.float32)
        self.ra.guard.previous=actual.copy()
        for i,item in enumerate(data['rows']):
            item['actual_action_after_physics_mask']=normalized[i].tolist()
            item['projection_changed']=bool(item['projection_changed'] or np.linalg.norm(normalized[i]-item['proposed_action'])>1e-5)
        nxt=self._packet()
        collision=bool(self.env.unwrapped.last_collision_records)
        reached=[]
        for i,raw in enumerate(data['goals']):
            g=next_goal(Goal.from_array(raw),nxt,i)
            event=bool(not collision and goal_reached(nxt,i,g,self.ra_options))
            reached.append(event)
            item=data['rows'][i]
            item.update(local_goal_reached=event,team_collision=collision)
            self.ra_counts[f'agent_{i}:goal_reached']+=event
            self.ra_counts[f'agent_{i}:{item["action_mode"]}']+=1
            self.ra_counts['agent_checks']+=1
        record=dict(packet=data['packet'],next_packet=nxt,goals=data['goals'].copy(),
            actual_actions=normalized,reward=float(np.mean(row['rewards'])),done=self.terminal,
            team_collision=collision,rows=data['rows'],reached=np.array(reached,bool))
        self.ra_records.append(record)
        row.update(reach_avoid=record,algorithm=SCHEMA)
        return row

    def summary(self):
        row=RepairedMissionRuntime.summary(self)
        trace=[x for record in self.ra_records for x in record['rows']]
        row.update(method='D2_ZERO_REACH_AVOID_V1',task_network_trained=False)
        row['reach_avoid']=dict(schema=SCHEMA,options=self.ra_options,behavior=self.behavior,
            ablation=self.ra.ablation,counts=dict(self.ra_counts),trace=trace,
            original_task_contract=[28,3,124],auxiliary_actor_input=INPUT_DIM,
            auxiliary_action_dim=3,public_directional_sectors=26,
            goal_condition='role/phase region, arrival before team collision within horizon',
            learned_values_calibrated=False,task_reward_changed=False,dynamics_changed=False,
            task_protocol_changed=False,hidden_obstacle_geometry_used_by_policy=False,
            old_policies_relabelled=False,safety_guarantee=False)
        return row
