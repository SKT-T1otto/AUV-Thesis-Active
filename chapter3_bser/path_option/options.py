"""Explicit architecture and comparison identities; parent options are untouched."""
import copy
import math

SCHEMA = 'auv.path_conditioned_options.v1'
VARIANTS = ('A', 'B', 'C', 'D', 'fixed_query', 'periodic', 'risk_only', 'rule_recovery')
INTERVENTIONS = ('none', 'no_path', 'no_history', 'rule_termination', 'rule_recovery', 'rule_action')
DEFAULTS = dict(schema=SCHEMA, history=16, width=64, layers=2, heads=4,
    horizon=32, candidates=8, gamma=.95, batch_size=32, replay_size=12000,
    warmup=128, update_every=4, actor_lr=.0003, critic_lr=.0005, tau=.01,
    route_query_interval=10, periodic_steps=100, minimum_commitment=10,
    min_option_seconds=2., max_option_seconds=30.,
    stagnation_steps=30, progress_distance=.25, arrival_radius=.75,
    reach_weight=1., risk_weight=1., entropy_weight=.001, seed=20261111)


def validate(value=None):
    o = copy.deepcopy(DEFAULTS if value is None else value)
    if set(o) != set(DEFAULTS) or o['schema'] != SCHEMA:
        raise ValueError('complete versioned path-option configuration required')
    for k, v in DEFAULTS.items():
        if k == 'schema':
            continue
        if isinstance(o[k], bool) or not math.isfinite(float(o[k])) or o[k] <= 0:
            raise ValueError('invalid option: ' + k)
        if type(v) is int and type(o[k]) is not int:
            raise ValueError('integer option required: ' + k)
    if o['gamma'] != .95 or o['horizon'] != 32 or o['candidates'] != 8:
        raise ValueError('discount/event/candidate identity changed')
    if o['width'] % o['heads'] or o['replay_size'] < o['warmup'] or o['warmup'] < o['batch_size']:
        raise ValueError('invalid architecture/replay sizes')
    return o
