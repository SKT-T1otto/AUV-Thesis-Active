"""One physical mission, four independent navigation instances, no team planner in control."""
import copy
import hashlib
import json
import numpy as np
import torch

from core.config.active_environment import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.env.task_protocol import PROTOCOL_FIELDS, validate_task_config
from chapter3_bser.integration.control_context import BSERControlContextV1, ExecutorAssignmentContextV1
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge, _mission_phase
from chapter3_bser.learning.base_runtime import seed_innovations
from chapter3_bser.protocols.team_environment import PRRACTrainingEnv
from chapter3_bser.rewards.training_env import Phase1CV2TrainingEnv
from chapter3_bser.safe_rl.dynamics import limits
from .navigation import LocalNavigator
from .observation import OnboardMap, sensor_frame


def joint_training_state(boats):
    rows = [boat.features for boat in boats]
    # The previous accepted root is available to centralized training. This is
    # taken before sampling ANY decision, and never made available to an actor.
    roots = []
    for boat in boats:
        task = boat.lifecycle.tasks.get(boat.agent)
        roots.append(np.zeros(3) if task is None or task.released else task.point / boat.bounds)
    return np.concatenate([*rows, *roots]).astype(np.float32)


def pack_guidance(items, task):
    # Only multiplex already selected local commands for the simulator API.
    ex = items[3]
    executor = ExecutorAssignmentContextV1(3, ex.assignment_kind, ex.final_waypoint, ex.planned_path,
                  ex.tracking_waypoint, ex.hold_position, ex.hold_state, ex.reachable, ex.execution_request)
    ids = [(item.assignment_id, tuple(x.item() if isinstance(x, np.generic) else x
                                     for x in item.final_waypoint), item.assignment_kind) for item in items]
    token = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return BSERControlContextV1(RMADDPGGuidanceBridge.SCHEMA_VERSION, 'local:' + token, token,
                task.step, _mission_phase(task), tuple(items), executor, 'ONBOARD')


class LocalMission:
    def __init__(self, config, options, scene):
        validate_task_config(config)
        self.horizon = config['max_steps']
        if scene['max_steps'] != self.horizon or scene['obstacle_knowledge_mode'] != 'online_unknown':
            raise ValueError('expected a native-horizon online-unknown scene')
        seed = int(scene['scenario_seed'])
        seed_innovations(seed)
        settings = build_ch3_config(config['base_candidate'], config['profile'])
        settings.update({key: config[key] for key in (*PROTOCOL_FIELDS, 'collision_terminal_reward',
                                                      'waypoint_reward_timing')})
        # Same external full-authority command interface as R123 SafetyRuntime.
        settings.update(use_residual_prior=False, residual_scale_search=1., residual_scale_executor=1.)
        self.core = MissionCoreEnv(**environment_kwargs_from_config(
            settings, device='cpu', max_steps=self.horizon, return_numpy=False))
        self.core.enable_public_event_capture('local_ctde', capacity=2048)
        guided = GuidedEnv(self.core, enabled=True)
        reward = Phase1CV2TrainingEnv(guided, reward_config=config['reward'])
        self.env = PRRACTrainingEnv(reward, reward_objective_config=config, gamma=config['rl']['gamma'])
        try:
            self.env.reset(scenario=scene)
            self.cursor = 0
            contract = self.core.get_public_control_contract()
            specs = self.core.get_public_vehicle_specs()
            self.scales = np.array([limits(spec) for spec in specs])
            self.boats = []
            for i in range(4):
                # This factory copies fixed configuration and constructs an empty
                # planner. It never copies cells or resets from physical obstacles.
                planner = self.core.unwrapped._build_obstacle_planner()
                view = OnboardMap(i, planner, specs)
                self.boats.append(LocalNavigator(
                    view, contract, options, self.horizon, scene['planner_obstacle_clearance'],
                    standby=scene['initial_executor_wait_point'] if i == 3 else None))
            seed_innovations(seed)
            self.receive()
        except BaseException:
            self.env.close()
            raise

    def receive(self):
        task = self.core.get_task_state()
        kin = self.core.get_agent_state()
        finished = self.core.get_search_execution_state().agent_finished
        events = self.core.read_public_events(self.cursor)
        delivery = self.core.get_public_delivered_target()
        for boat in self.boats:
            frame = sensor_frame(boat.agent, task.step, kin, finished, events, delivery)
            boat.sense(frame)
        if events:
            self.cursor = events[-1].sequence
        return task, delivery

    def rollout(self, actor, generators, cap, *, log_training=False, departure=None, observe=None):
        if not 1 <= cap <= self.horizon:
            raise ValueError('invalid rollout cap')
        policies = [copy.deepcopy(actor).eval() for _ in range(3)]
        rewards, states, decisions, frames = [], [], [], []
        first_found = first_delivery = None
        max_action_error = 0.
        initial = self.core.get_agent_state().positions
        if departure is not None and (log_training or self.core.get_task_state().step != 0):
            raise ValueError('a departure intervention requires a fresh evaluation episode')
        for tick in range(cap):
            if log_training:
                states.append(joint_training_state(self.boats))
            commands, items = [], []
            if departure is not None and tick == 0:
                actions = departure(self.boats, policies, generators)
            else:
                actions = [boat.act(policies[i] if i < 3 else None, generators[i] if i < 3 else None)
                           for i, boat in enumerate(self.boats)]
            for command, item, draw in actions:
                if draw is not None:
                    decisions.append(draw)
                commands.append(command)
                items.append(item)
            self.env.install_guidance(pack_guidance(items, self.core.get_task_state()))
            _, reward, dones = self.env.step(torch.from_numpy(np.stack(commands)))
            team = float(self.env.reward_accounting.last['team_reward'])
            if not np.allclose(np.asarray(reward), team, rtol=0, atol=1e-6):
                raise RuntimeError('original shared team reward disagrees across boats')
            rewards.append(team)
            actual = self.core.unwrapped._agent_acc.detach().cpu().numpy().copy()
            for boat in self.boats:
                boat.own_applied_control(actual[boat.agent])
            error = np.abs(actual / self.scales - np.array(commands)).max()
            max_action_error = max(max_action_error, float(error))
            event_cursor = self.cursor
            task, delivery = self.receive()  # also read terminal-step handoff
            if task.target_found and first_found is None:
                first_found = self.core.unwrapped.found_step
            if delivery is not None and first_delivery is None:
                first_delivery = delivery['received_step']
            result = self.core.get_episode_result()
            if observe is not None:
                observe(self, event_cursor, task, delivery, result, team)
            if task.step != len(rewards) or list(map(bool, dones)) != [result['terminated']] * 4:
                raise RuntimeError('mission clock or terminal contract mismatch')
            frames.append(dict(step=task.step, positions=[list(b.env.frame.position) for b in self.boats],
                               commands=np.asarray(commands).tolist(), applied_acceleration=actual.tolist(),
                               routes=[b.manager.current[b.agent].route.route_id for b in self.boats],
                               roots=[b.lifecycle.tasks[b.agent].point.tolist() for b in self.boats],
                               executor_mode=items[3].assignment_kind))
            if result['terminated']:
                break
        if max_action_error > 1e-6:
            raise RuntimeError('joint simulator altered a local physical command')
        out = dict(rewards=rewards, decisions=decisions, frames=frames, initial_positions=initial,
                   steps=len(rewards), found_step=first_found, delivery_step=first_delivery,
                   finder_id=task.finder_id if task.target_found else None,
                   terminated=bool(result['terminated']), success=bool(task.mission_complete) if result['terminated'] else None,
                   termination_reason=result['termination_reason'] if result['terminated'] else 'observation_cap',
                   team_return=sum(rewards), max_action_error=max_action_error,
                   scan_counts=[b.env.scan_count for b in self.boats],
                   projection_queries=sum(b.proposal_queries for b in self.boats),
                   execution_route_requests=sum(b.sync_route_queries for b in self.boats))
        if log_training:
            out.update(states=states, final_state=joint_training_state(self.boats))
        return out

    def close(self):
        self.env.close()
