import itertools
import math

def paired_ordering(rows):
    groups = {}
    identity = ('scenario_id', 'scenario_seed', 'scenario_sha256', 'environment_innovation_seed', 'gamma', 'task_protocol', 'reward_objective', 'source_reward_revision', 'waypoint_reward_timing', 'max_steps')
    required = (*identity, 'policy_id', 'trajectory_complete', 'team_discounted_return', 'found_step', 'pre_found_collision', 'found_penalized_steps')
    for row in rows:
        if not all((key in row for key in required)):
            raise ValueError('ordering requires complete, explicitly identified paired evaluation rows')
        if row['trajectory_complete'] is not True:
            raise ValueError('incomplete missions cannot be ranked as failures')
        for key in ('team_discounted_return', 'found_penalized_steps'):
            if not math.isfinite(row[key]):
                raise ValueError('nonfinite ordering metric')
        key = tuple((row[k] for k in identity))
        group = groups.setdefault(key, {})
        if row['policy_id'] in group:
            raise ValueError('duplicate policy/scenario evaluation row')
        group[row['policy_id']] = row
    result = {metric: dict(comparable_pairs=0, discordant_pairs=0, metric_ties=0) for metric in ('found', 'pre_found_collision', 'found_penalized_steps')}
    examples = []
    for group in groups.values():
        for (a, b) in itertools.combinations(group.values(), 2):
            delta_return = a['team_discounted_return'] - b['team_discounted_return']
            if delta_return == 0:
                continue
            (better, worse) = (a, b) if delta_return > 0 else (b, a)
            deltas = dict(found=int(better['found_step'] is not None) - int(worse['found_step'] is not None), pre_found_collision=int(worse['pre_found_collision']) - int(better['pre_found_collision']), found_penalized_steps=worse['found_penalized_steps'] - better['found_penalized_steps'])
            for (metric, delta) in deltas.items():
                result[metric]['metric_ties'] += int(delta == 0)
                result[metric]['comparable_pairs'] += int(delta != 0)
                result[metric]['discordant_pairs'] += int(delta < 0)
                if delta < 0 and len(examples) < 20:
                    examples.append(dict(metric=metric, scenario_id=better['scenario_id'], higher_return_policy=better['policy_id'], lower_return_policy=worse['policy_id']))
    return dict(metrics=result, examples=examples, paired_scenario_groups=sum((len(g) > 1 for g in groups.values())), reward_metric_equivalence_established=False, inference='descriptive paired ordering only')
