"""Independent OLD algorithm, using unchanged physical integrator and rewards."""
import copy
import numpy as np
import torch
from core.config.active_environment import build_ch3_config
from core.env.mission_env import MissionCoreEnv,environment_kwargs_from_config
from core.env.task_protocol import PROTOCOL_FIELDS,validate_task_config
from core.mapping.planning_state import extract_planning_state
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.safe_rl.options import validate,SCHEMA
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.navigation import SafetyNavigationBridge
from chapter3_bser.safe_rl.dynamics import limits,speed_audit


class SafetyRuntime(RepairedMissionRuntime):
    def __init__(self,config,scenario,*,seed,episode_id=0,options=None,learner=None,collection_only=False):
        self.safety_options=validate(options)
        self.collection_only=bool(collection_only)
        if config.get('map_experiment',{}).get('map_id','OLD')!='OLD' or config.get('map_experiment',{}).get('research_config') is not None:
            raise ValueError('safety RL v2 requires OLD research_config=None')
        guard=config.get('map_experiment',{}).get('executor_safety')
        if guard is not None and guard.get('enabled'):raise ValueError('do not combine v1/v2 executor guards with all-agent safety RL')
        self._safety_episode_id=episode_id
        super().__init__(config,scenario,seed=seed,episode_id=episode_id,map_research=None)
        self.nominal_contract=dict(self.env.get_public_control_contract(),**self._original_composition)
        self.safety_controller=SafetyController(self.env.get_public_vehicle_specs(),self.nominal_contract,self.safety_options,learner=learner)
        self.safety_bridge=SafetyNavigationBridge(self.bridge,self.safety_controller,self.env,self.safety_options)
        if not self.collection_only:self.bridge=self.safety_bridge
        self.safety_transitions=[]
        self._last_safety=None

    def _make_environment(self,config):
        validate_task_config(config)
        settings=build_ch3_config(str(config.get('base_candidate','ch3_v3_full_reference')),str(config['profile']))
        settings.update({k:config[k] for k in (*PROTOCOL_FIELDS,'collision_terminal_reward') if k in config})
        if 'waypoint_reward_timing' in config:settings['waypoint_reward_timing']=config['waypoint_reward_timing']
        self._original_composition={k:settings[k] for k in ('use_residual_prior','residual_scale_search','residual_scale_executor')}
        # Explicit NEW input-composition contract, not modified physical dynamics.
        # Compose original prior/residual externally, then permit a full-authority
        # safety replacement through the existing normalized acceleration input.
        settings.update(use_residual_prior=False,residual_scale_search=1.,residual_scale_executor=1.)
        base=MissionCoreEnv(**environment_kwargs_from_config(settings,device='cpu',max_steps=int(config['max_steps']),return_numpy=False))
        base.enable_public_event_capture('safe_rl:'+str(self._safety_episode_id),capacity=2048)
        return base

    def _actions_for_step(self,actions):
        task=super()._actions_for_step(actions).numpy().copy()
        state=extract_planning_state(self.env);controller=self.safety_controller
        commands,features,rows,geometry=controller.act(state,self.guidance,task,
            self.env.read_public_events(controller.evidence.cursor),
            semantic_goals=self.safety_bridge.semantic_goals or None,
            cache_age=self.step-self.provider._last_full_refresh_step)
        if self.collection_only:
            scales=np.array([limits(s) for s in controller.specs])
            commands=np.asarray([r['nominal_acceleration'] for r in rows],np.float32)/scales
            for i,r in enumerate(rows):
                r.update(mode='NOMINAL_COLLECTION',model_checked=False,executed_acceleration=r['nominal_acceleration'],
                    safety_filter_bypassed_for_factual_original_D2_data=True)
            controller.previous=np.asarray([r['nominal_acceleration'] for r in rows])
        self._last_safety=dict(safety_features=features,safety_rows=rows,task_residual_actions=task,
            normalized_executed_acceleration=commands.copy())
        return torch.as_tensor(commands,dtype=torch.float32)

    def advance(self,model,*,explore=True):
        row=super().advance(model,explore=explore)
        data=self._last_safety
        # Record actual post-clip/post-inactive-mask control, not an ignored proposal.
        actual=self.env.unwrapped._agent_acc.detach().cpu().numpy().copy()
        scales=np.array([limits(s) for s in self.safety_controller.specs])
        data['normalized_executed_acceleration']=(actual/scales).astype(np.float32)
        costs=np.zeros(4,np.float32)
        for hit in self.env.unwrapped.last_collision_records:costs[int(hit['agent_id'])]=1.
        data['collision_cost']=costs
        for i,item in enumerate(data['safety_rows']):
            item['actual_acceleration_after_physics_mask']=actual[i].tolist()
        row.update(data,safety_algorithm=SCHEMA,task_actions=row.get('proposed_actions'),
            physical_action_contract='normalized_absolute_acceleration_v1')
        self.safety_transitions.append(copy.deepcopy(data))
        return row

    def summary(self):
        row=super().summary();c=self.safety_controller
        row.update(method='D2_ZERO_nominal_collection_v2' if self.collection_only else 'D2_ALL_AGENT_SAFE_RL_v2' if c.learner is not None else 'D2_ALL_AGENT_RULE_SAFETY_v2',
            task_reference='D2_ZERO',task_network_trained=False)
        row['safety_rl']=dict(schema=SCHEMA,options=self.safety_options,counts=dict(c.counts),
            navigation_counts=dict(self.safety_bridge.counts),navigation_trace=list(self.safety_bridge.trace),
            trace=list(c.trace),speed_audit=speed_audit(c.specs,c.contract,self.safety_options),
            public_scan_count=c.evidence.scans,retained_hit_points=len(c.evidence.hits),hit_evictions=c.evidence.evictions,
            original_task_contract=[28,3,124],safety_contract=[48*self.safety_options['history'],3,self.safety_options['horizon']],
            physical_action_contract='normalized_absolute_acceleration_v1',
            task_reward_changed=False,dynamics_changed=False,ground_truth_geometry_used_by_safety=False,
            original_task_observation_has_truth_derived_range=True,unprotected_nominal_collection=self.collection_only,safety_guarantee=False)
        return row
