"""Independent algorithm identity, physical assumptions and learning contract."""
import copy
import math

SCHEMA = 'auv.invalid_goal_safe_rl.v1'
FRAME_DIM = 48
DEFAULTS = dict(schema=SCHEMA, history=4, hidden_dim=128, horizon=32,
    clearance=.4, hit_padding=.12, flow_component_bound=.18, reaction_steps=1,
    uncertainty_slowdown=.35, risk_threshold=.15, release_steps=3,
    repair_interval=5, reject_after=3, quarantine_steps=40,
    actor_lr=.001, critic_lr=.0005, batch_size=128, replay_size=50000,
    warmup_samples=256, update_frequency=4, minimum_updates=32,
    recovery_imitation_weight=.5, seed=20261027)


def validate(value=None):
    c=copy.deepcopy(DEFAULTS if value is None else value)
    if set(c)!=set(DEFAULTS) or c['schema']!=SCHEMA:
        raise ValueError('complete versioned safety-RL options required')
    for key in ('history','hidden_dim','horizon','reaction_steps','release_steps','repair_interval',
                'reject_after','quarantine_steps','batch_size','replay_size','warmup_samples',
                'update_frequency','minimum_updates','seed'):
        if type(c[key]) is not int or c[key]<1:raise ValueError('invalid safety integer: '+key)
    if c['history']!=4 or c['hidden_dim']!=128 or c['horizon']!=32:
        raise ValueError('v1 safety network contract is 4 x 48, hidden128, horizon32')
    for key in set(c)-{'schema'}-{k for k,v in DEFAULTS.items() if type(v) is int}:
        if isinstance(c[key],bool) or not math.isfinite(float(c[key])) or float(c[key])<=0:
            raise ValueError('invalid safety scalar: '+key)
    if c['flow_component_bound']<.18 or c['reaction_steps']<1:
        raise ValueError('must cover the unchanged OLD flow and one-step reaction')
    if not 0<c['risk_threshold']<1 or not 0<c['uncertainty_slowdown']<1:
        raise ValueError('invalid probability or confidence slowdown')
    if c['replay_size']<c['batch_size'] or c['warmup_samples']<c['batch_size']:
        raise ValueError('insufficient independent safety replay capacity/warmup')
    return c
