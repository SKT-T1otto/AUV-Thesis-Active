"""New algorithm identity; none of these options modify the parent references."""
import copy
import math

SCHEMA = 'auv.goal_conditioned_reach_avoid.v1'
FRAME_DIM = 90
HISTORY = 4
GOAL_DIM = 19
INPUT_DIM = FRAME_DIM * HISTORY + GOAL_DIM
DEFAULTS = dict(schema=SCHEMA, history=4, hidden_dim=128, horizon=32,
    candidates=8, action_bank_size=16, goal_radius=.75, capture_radius=.8,
    hold_speed=.2, clearance=.4, hit_padding=.12, flow_bound=.18,
    gamma=.95, actor_lr=.001, critic_lr=.0005, target_tau=.01,
    batch_size=128, replay_size=30000, warmup_samples=256, minimum_updates=32,
    update_frequency=4, reach_floor=.2, risk_limit=.15, reach_weight=1.,
    dual_lr=.01, dual_max=20., projection_weight=.1, hindsight_per_agent=1,
    seed=20261027)

# Public, frozen original vehicle/model contract. Runtime checks equality against
# the public APIs before using it. No flow phase or obstacle truth is included.
SPECS = (
    dict(a_xy_max=1.3, a_z_max=.75, v_xy_max=2.8, v_z_max=1.2, drag_xy=.1, drag_z=.16, buoyancy_bias=0.),
    dict(a_xy_max=1., a_z_max=.7, v_xy_max=2.2, v_z_max=1., drag_xy=.12, drag_z=.18, buoyancy_bias=0.),
    dict(a_xy_max=.9, a_z_max=.65, v_xy_max=1.8, v_z_max=.9, drag_xy=.14, drag_z=.2, buoyancy_bias=0.),
    dict(a_xy_max=.8, a_z_max=.55, v_xy_max=1.5, v_z_max=.75, drag_xy=.16, drag_z=.24, buoyancy_bias=.02),
)
CONTRACT = dict(dt=.2, eps=1e-6, prior_kv_xy=1.1, prior_kv_z=1.,
    prior_slow_radius_xy=2.4, prior_slow_radius_z=1.2)


def validate(value=None):
    result = copy.deepcopy(DEFAULTS if value is None else value)
    if set(result) != set(DEFAULTS) or result['schema'] != SCHEMA:
        raise ValueError('complete reach-avoid v1 options required')
    for key, default in DEFAULTS.items():
        if key == 'schema':
            continue
        x = result[key]
        if isinstance(x, bool) or not math.isfinite(float(x)) or x <= 0:
            raise ValueError('invalid reach-avoid option: ' + key)
        if type(default) is int and type(x) is not int:
            raise ValueError('integer reach-avoid option required: ' + key)
    for key in ('history', 'hidden_dim', 'horizon', 'candidates', 'action_bank_size', 'gamma', 'clearance', 'hit_padding', 'flow_bound'):
        if result[key] != DEFAULTS[key]:
            raise ValueError('versioned reach-avoid architecture/discount changed: ' + key)
    if result['replay_size'] < 3 * result['batch_size'] or result['warmup_samples'] < result['batch_size']:
        raise ValueError('insufficient stratified replay capacity')
    if not 0 < result['target_tau'] <= 1 or not 0 < result['reach_floor'] < 1 or not 0 < result['risk_limit'] < 1:
        raise ValueError('invalid reach-avoid probability/target rate')
    if result['flow_bound'] < .18:
        raise ValueError('must cover original public flow bound')
    return result
