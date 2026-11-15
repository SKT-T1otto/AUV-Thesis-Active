"""Frozen arms: exactly one requested mechanism is withdrawn in each ablation."""
import copy
from chapter3_bser.path_option.options import DEFAULTS, validate as learning_options

SCHEMA = 'auv.task_progress.r123.v1'
ARMS = ('full', 'no_r1', 'no_r2', 'no_r3')
CONTROL = dict(residual_bound=.15, stagnation_steps=30, progress_distance=.025,
               max_recoveries=2, recovery_budget_steps=150, request_interval=10)


def switches(arm):
    if arm not in (*ARMS, 'rule'):
        raise ValueError('unknown R123 arm')
    return dict(r1=arm != 'no_r1', r2=arm != 'no_r2', r3=arm != 'no_r3')


def options(value=None):
    result = copy.deepcopy(DEFAULTS if value is None else value)
    return learning_options(result)
