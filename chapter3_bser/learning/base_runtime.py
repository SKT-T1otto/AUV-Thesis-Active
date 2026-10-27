from __future__ import annotations
import copy
import random
import numpy as np
import torch
from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.rewards.training_env import Phase1CV2TrainingEnv
from chapter3_bser.protocols.controller_factory import build_prrac_online_controller
from chapter3_bser.protocols.team_environment import PRRACTrainingEnv
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.online.config import execution_runtime_config
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.config.active_environment import build_ch3_config
from core.env.mission_env import MissionCoreEnv
from core.env.mission_env import environment_kwargs_from_config
from core.env.task_protocol import PROTOCOL_FIELDS
from core.env.task_protocol import strict_terminal
from core.env.task_protocol import validate_task_config

def seed_innovations(seed):
    """Keep the reference innovation-seeding convention without importing HGR."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(np.random.SeedSequence(seed).generate_state(4))
    torch.manual_seed(seed)

def _make_base_env(config):
    validate_task_config(config)
    settings = build_ch3_config(str(config.get('base_candidate', 'ch3_v3_full_reference')), str(config['profile']))
    settings.update({key: config[key] for key in (*PROTOCOL_FIELDS, 'collision_terminal_reward') if key in config})
    if 'waypoint_reward_timing' in config:
        settings['waypoint_reward_timing'] = config['waypoint_reward_timing']
    return MissionCoreEnv(**environment_kwargs_from_config(settings, device='cpu', max_steps=int(config['max_steps']), return_numpy=False))

def _public_context(env, state):
    return OnlineMissionContext.from_public_views(env.get_task_state(), env.get_search_execution_state(), state)

def _observations(values):
    result = [torch.as_tensor(value).detach().cpu().numpy().copy() for value in values]
    array = np.asarray(result)
    if array.shape != (4, 28) or not np.isfinite(array).all():
        raise ValueError('baseline runtime requires finite 4 x 28 observations')
    return result

class BaselineMissionRuntime:
    """Use the existing BSER, physical action prior and final reward adapters.

    B2/B3 learn direct actor outputs under the same environment action contract.
    They do not contain a learned expert or action residual mixing network; the
    shared environment's prior/physical action mapping remains unchanged.
    """

    def __init__(self, config, scenario, *, seed, episode_id=0, map_research=None):
        self.config = copy.deepcopy(config)
        self.scenario = copy.deepcopy(scenario)
        import chapter3_bser.planner_factory as assembly
        from chapter3_bser.d2.contract import enabled
        assembly.prepare(self, config, scenario, map_research=map_research)
        seed_innovations(seed)
        base = self._make_environment(config)
        assembly.before_reset(self, base, episode_id)
        guided = GuidedEnv(base, enabled=True)
        v2 = Phase1CV2TrainingEnv(guided, reward_config=config['reward'])
        self.env = PRRACTrainingEnv(v2, reward_objective_config=config, gamma=config['rl']['gamma'])
        self.found_step = None
        self.handoff_event_step = None
        self.handoff_decision_step = None
        try:
            self.env.reset(scenario=scenario, episode_id=episode_id, episode_index=episode_id)
            assembly.consume(self)
            seed_innovations(seed)
            phase_config = load_phase1b2_config()
            runtime_config = execution_runtime_config(config)
            phase_config['execution_runtime'] = copy.deepcopy(runtime_config)
            self.provider = OnlinePlanningStateProvider(self.env, refresh_interval=int(phase_config['online']['state_refresh_interval']), **{key: runtime_config[key] for key in ('refresh_on_executor_handoff', 'refresh_on_public_target_shift', 'public_target_update_distance', 'public_target_update_min_steps')})
            self.state = self.provider.initialize()
            context = _public_context(self.env, self.state)
            self.controller = build_prrac_online_controller(phase_config, config)
            if enabled(config):
                self.controller = assembly.build_controller(self, self.controller)
            initialized = self.controller.initialize(self.state, context)
            self.bridge = assembly.build_bridge(self) if enabled(config) else RMADDPGGuidanceBridge()
            self.guidance = self.bridge.compile_guidance(initialized.allocation, self.state, context, decision_reason='INITIALIZE')
            self.env.install_guidance(self.guidance)
            self.observations = self.env.refresh_observation_after_guidance()
            assembly.installed(self)
            _observations(self.observations)
            self._record_events()
        except Exception:
            self.env.close()
            raise

    @staticmethod
    def _make_environment(config):
        return _make_base_env(config)

    @property
    def step(self):
        return int(self.env.get_task_state().step)

    @property
    def terminal(self):
        return strict_terminal(self.env)

    def _record_events(self):
        task = self.env.get_task_state()
        if task.target_found and self.found_step is None:
            self.found_step = int(self.env.unwrapped.found_step)
        if task.executor_knows_target and self.handoff_event_step is None:
            self.handoff_event_step = task.handoff_step
        known = bool(self.env.get_search_execution_state().target_known_by_agent[3])
        if known and (not self.terminal) and (self.handoff_decision_step is None):
            self.handoff_decision_step = self.step

    def advance(self, model, *, explore=True):
        if self.terminal:
            raise RuntimeError('cannot step a terminal mission')
        before = self.step
        observations = _observations(self.observations)
        actions = torch.cat(model.step(observations, explore=explore), dim=0).detach().cpu()
        if actions.shape != (4, 3) or not bool(torch.isfinite(actions).all()):
            raise ValueError('baseline actor must produce finite 4 x 3 actions')
        actions = self._actions_for_step(actions)
        (step_obs, rewards, dones) = self.env.step(actions)
        import chapter3_bser.planner_factory as assembly
        assembly.consume(self)
        if self.terminal:
            self.observations = step_obs
        else:
            self.state = self.provider.snapshot(force=False)
            context = _public_context(self.env, self.state)
            result = self.controller.step(self.state, context)
            self.env.observe_controller_result(result, controller=self.controller, state_provider=self.provider)
            self.guidance = self.bridge.compile_guidance(result.allocation, self.state, context, decision_reason=result.decision_reason)
            self.env.install_guidance(self.guidance)
            self.observations = self.env.refresh_observation_after_guidance()
            assembly.installed(self)
        self._record_events()
        done_values = [bool(done) for done in dones]
        if len(done_values) != 4 or self.step != before + 1 or self.terminal != all(done_values):
            raise RuntimeError('baseline mission clock/done contract mismatch')
        return dict(t=before, observations=observations, actions=actions.numpy().copy(), rewards=torch.as_tensor(rewards).detach().cpu().numpy().copy(), dones=done_values, next_observations=_observations(self.observations), **copy.deepcopy(self.env.reward_accounting.last))

    def summary(self):
        summary = self.env.finalize_episode()
        import chapter3_bser.planner_factory as assembly
        from chapter3_bser.d2.contract import enabled
        if enabled(self.config):
            summary['planner'] = assembly.diagnostics(self)
        summary.update(scenario_id=self.scenario['scenario_id'], scenario_seed=self.scenario['scenario_seed'], actual_length=self.step, found_step=self.found_step, handoff_event_step=self.handoff_event_step, handoff_decision_step=self.handoff_decision_step, remaining_task_steps=int(self.config['max_steps']) - self.step)
        return summary

    def close(self):
        self.env.close()
