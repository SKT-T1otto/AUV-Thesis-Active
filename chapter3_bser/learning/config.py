import copy
import math
import numpy as np
from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.checkpoint import validate_config
VERSION = 'ch3.d2.implementation_repair.v2'
PARENTS = {'D2_B2_repair_v2': ('B2_direct_mc', 'maddpg', 'ch3_baseline_direct_mc', 'ch3.baseline.maddpg.v1'), 'D2_B3_repair_v2': ('B3_direct_boundary', 'direct_boundary_maddpg', 'ch3_baseline_direct_boundary', 'ch3.baseline.boundary_maddpg.v1')}
FLAGS = ('waypoint_arrival', 'inactive_actor', 'hold_residual')

def scenario_fingerprint(scene):
    """Hash consumed physical inputs, never derived checksums or annotations.

    Identity and innovation seed checks remain separate. Planner seeds are not
    physical initial conditions. This identifies repeated initial tasks, not
    equivalent stochastic trajectories. The field list is
    tied to MissionCoreEnv.reset and SafeSearch's planner clearance input.
    """

    def vector(value, shape, *, dtype=np.float32):
        raw = np.asarray(value, dtype=np.float64)
        if raw.shape != shape or not np.isfinite(raw).all():
            raise ValueError('invalid physical fingerprint vector')
        with np.errstate(over='ignore', under='ignore'):
            converted = raw.astype(dtype)
        if not np.isfinite(converted).all():
            raise ValueError('physical fingerprint vector overflows environment dtype')
        converted[converted == 0] = 0
        return converted.tolist()

    def scalar(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('nonfinite physical fingerprint scalar')
        return 0.0 if value == 0 else value
    layout = scene.get('obstacle_layout_id', 'none')
    obstacles = []
    if layout == 'custom_aabb_v1':
        obstacles = [dict(center=vector(o['center'], (3,)), size=vector(o['size'], (3,))) for o in scene.get('obstacles', [])]
        obstacles.sort(key=digest)
    physical = dict(schema='ch3.d2.physical_input.v2', initial_agent_positions=vector(scene.get('initial_agent_positions'), (4, 3)), initial_executor_wait_point=vector(scene.get('initial_executor_wait_point'), (3,)), target_position=vector(scene.get('target_initial_position', scene.get('target_position')), (3,)), target_initial_velocity=vector(scene.get('target_initial_velocity', [0, 0, 0]), (3,), dtype=np.float64), target_motion_mode=scene.get('target_motion_mode', 'constant_velocity_reflect_v1'), obstacle_layout_id=layout, obstacle_knowledge_mode=scene.get('obstacle_knowledge_mode', 'online_unknown'), obstacles=obstacles, flow_phase_x=scalar(scene.get('flow_phase_x', 0)), flow_phase_y=scalar(scene.get('flow_phase_y', 0)), planner_obstacle_clearance=scalar(scene.get('planner_obstacle_clearance', 0.4)))
    return digest(physical)

def validate(config):
    c = copy.deepcopy(config)
    if 'map_experiment' in c:
        from chapter3_bser.learning.map_contract import validate_condition
        c['map_experiment']=validate_condition(c['map_experiment'])
    if 'scenario_schedule' in c:
        order=c['scenario_schedule']
        if (not isinstance(order,list) or len(order)!=c.get('total_main_trajectories')
                or any(type(i) is not int or i<0 for i in order) or not c.get('scenario_manifest')):
            raise ValueError('explicit episode schedule requires one valid manifest index per training episode')
    if c.get('schema') != VERSION or c.get('baseline') not in PARENTS:
        raise ValueError('requires an independent D2 repair configuration')
    (baseline, algorithm, method, architecture) = PARENTS[c['baseline']]
    if c.get('method') != c['baseline'] or c.get('algorithm') != algorithm or c.get('architecture_version') != architecture or (c.get('checkpoint_schema') != VERSION):
        raise ValueError('repair method/architecture identity mismatch')
    flags = c.get('repairs', {})
    if set(flags) != set(FLAGS) or any((type(flags[k]) is not bool for k in FLAGS)):
        raise ValueError('three explicit boolean repair switches are required')
    parent = copy.deepcopy(c)
    parent.update(schema='ch3.baseline.training.v1', baseline=baseline, method=method, checkpoint_schema='ch3.baseline.maddpg.v1')
    validate_config(parent, baseline)
    if c['rl']['gamma'] != 0.95 or c.get('planner_protocol') != 'd2_v1':
        raise ValueError('repair v2 retains gamma=.95 and the D2 planner')
    if c.get('performance', {}).get('learner_device', 'cpu') != 'cpu' or c.get('performance', {}).get('cpu_threads', 1) != 1:
        raise ValueError('repair v2 currently uses the reference CPU learner with one thread')
    expected = {'schema': 'bser.phase1c.execution_reward.v2', 'enabled': True, 'freeze_searchers_after_found': True, 'preserve_discovery_reward': True, 'contact_entry_bonus': 0.25, 'hold_increment_bonus': 0.2, 'terminal_success_bonus': 2.0, 'reward_clip': 3.0, 'executor_id': 3, 'searcher_ids': [0, 1, 2]}
    if c['reward'] != expected:
        raise ValueError('repair v2 preserves the original composite reward settings')
    if c.get('collision_terminal_reward') != -2.0:
        raise ValueError('repair v2 preserves the original collision reward -2')
    timing = 'arrival_before_reward_v1' if flags['waypoint_arrival'] else 'legacy_v1'
    if c.get('waypoint_reward_timing', timing) != timing:
        raise ValueError('conflicting waypoint repair revision')
    c['waypoint_reward_timing'] = timing
    return c
