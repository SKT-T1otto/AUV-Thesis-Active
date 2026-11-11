"""Task, mechanism, learning and runtime evidence are reported separately."""
import numpy as np


def episode(row, maximum):
    trace = row.get('path_option', {}).get('trace', [])
    active = [x for x in trace if x['phase'] != 3]
    collision = row['termination_reason'] == 'obstacle_collision'
    success = bool(row['success'])
    if success != (row['termination_reason'] == 'success'):
        raise ValueError('inconsistent termination')
    ids = row.get('first_collision_agent_ids', [])
    switches = sum(x['task_switch'] for x in active)
    forced = sum(x['forced'] >= 0 for x in active)
    recoveries = {}
    longest = [0]*4; runs = [0]*4
    for x in active:
        slow = np.linalg.norm(x['velocity']) < .05 and np.linalg.norm(np.array(x['endpoint'])-x['position']) > .8
        runs[x['agent']] = runs[x['agent']]+1 if slow else 0
        longest[x['agent']] = max(longest[x['agent']], runs[x['agent']])
        if x['mode'] == 'RECOVER':
            recoveries.setdefault((x['agent'], x['goal_id']), []).append(x)
    arrivals = sum(any(x.get('endpoint_arrived', False) for x in xs) for xs in recoveries.values())
    progress = [float(np.linalg.norm(np.array(x['position'])-x['task'])-
        np.linalg.norm(np.array(x['next_position'])-x['task'])) for xs in recoveries.values() for x in xs]
    return dict(scenario_id=row['scenario_id'], success=int(success), collision=int(collision),
        timeout=int(row['termination_reason'] == 'timeout'), found=int(bool(row.get('found', False))),
        steps=row['actual_length'], penalized_steps=row['actual_length'] if success else maximum,
        team_discounted_return=float(row['team_discounted_return']),
        executor_search_collision=int(3 in ids and row.get('first_collision_phase') == 'Search'),
        executor_other_collision=int(3 in ids and row.get('first_collision_phase') != 'Search'),
        searcher_collision=int(any(i in ids for i in (0, 1, 2))),
        task_switches=switches, option_terminations=sum(x['decision']['terminate'] for x in active),
        forced_fraction=forced/max(1, len(active)), emergency_fraction=sum(x['emergency'] for x in active)/max(1, len(active)),
        policy_executed_fraction=sum(x['policy_executed'] for x in active)/max(1, len(active)),
        mean_tracking_error=float(np.mean([x['path_tracking_error'] for x in active])) if active else 0.,
        stalled_fraction=sum(x['reason'] == 'STALLED' for x in active)/max(1, len(active)),
        longest_stall_steps=max(longest), recovery_attempts=len(recoveries), recovery_region_arrivals=arrivals,
        recovery_unfinished=len(recoveries)-arrivals,
        mean_recovery_observed_steps=float(np.mean([len(xs) for xs in recoveries.values()])) if recoveries else 0.,
        recovery_then_episode_success=int(bool(recoveries) and success),
        recovery_task_distance_progress=float(np.sum(progress)),
        recovery_agent_steps=sum(x['mode'] == 'RECOVER' for x in active))


def aggregate(rows):
    if not rows:
        raise ValueError('no completed episodes')
    result = dict(episodes=len(rows))
    for k in rows[0]:
        if k != 'scenario_id':
            result[k] = float(np.mean([r[k] for r in rows]))
    return result


def learning_assessment(curve):
    if len(curve) < 4:
        return dict(status='insufficient_evaluations', convergence_proved=False)
    tail = curve[-3:]
    spread = {k: float(np.ptp([r[k] for r in tail])) for k in ('success', 'collision', 'timeout', 'penalized_steps')}
    return_available = all('team_discounted_return' in r for r in curve)
    return_spread = float(np.ptp([r['team_discounted_return'] for r in tail])) if return_available else None
    return_stable = return_spread <= .05*max(1., abs(curve[0]['team_discounted_return'])) if return_available else None
    stable = spread['success'] <= .05 and spread['collision'] <= .05 and spread['timeout'] <= .05 and spread['penalized_steps'] <= 20 and return_stable is not False
    good = np.mean([r['success'] for r in tail]) >= .8 and np.mean([r['timeout'] for r in tail]) <= .2 and np.mean([r['collision'] for r in tail]) <= .1
    mean = lambda k: float(np.mean([r[k] for r in tail]))
    gain = mean('success')-curve[0]['success'] >= 1/max(1, curve[0].get('episodes', 12))-1e-9 or (mean('success') >= curve[0]['success']-.05 and curve[0]['penalized_steps']-mean('penalized_steps') > 20)
    updates = curve[-1].get('updates', 0) > curve[0].get('updates', 0)
    status = 'still_changing' if not stable else 'stable_but_ineffective' if not good else 'empirically_stable_candidate' if gain and updates else 'stable_good_policy_without_clear_learning_gain'
    return dict(status=status, task_quality_passed=bool(good), return_stable=return_stable,
        return_tail_range=return_spread, online_task_gain=bool(gain), learning_updates_observed=bool(updates),
        empirical_plateau=bool(stable), convergence_proved=False, independent_test_acceptance=False, tail_ranges=spread)


def event_diagnostics(rows, horizon=32):
    """ONLY factual behavior continuations with unchanged option/event references.

    Switched/censored examples are excluded and counted. This is NOT evaluation
    of a hypothetical fixed-option policy or a probability calibration claim.
    """
    samples = []; censored = 0
    for agent in range(4):
        stream = [x for x in rows if x['agent'] == agent and x['phase'] != 3]
        for j, x in enumerate(stream):
            if x['decision']['reach'] is None:
                continue
            observed = None
            for k, y in enumerate(stream[j:j+horizon]):
                if k and (y['goal_id'] != x['goal_id'] or y['phase'] != x['phase']):
                    break
                if y['collision']:
                    observed = 0; break
                # A waypoint change means this log no longer labels the original query.
                if k and y.get('event_target') != x.get('event_target'):
                    break
                if y['reached']:
                    observed = 1; break
                if k == horizon-1:
                    observed = 0
            if observed is None:
                censored += 1
            else:
                samples.append((x['decision']['reach'], observed))
    return dict(scope='factual behavior continuation diagnostic; censoring is selective',
        hypothetical_policy_replay=False, calibrated_probability_claimed=False,
        labeled=len(samples), censored=censored,
        brier=float(np.mean([(p-y)**2 for p, y in samples])) if samples else None)
