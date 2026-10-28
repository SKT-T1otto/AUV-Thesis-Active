"""Predeclared empirical learning diagnostics, never a convergence theorem."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SCHEMA = 'auv.reach_avoid.monitoring.v1'
DEFAULTS = dict(schema=SCHEMA, eval_every=50, tail_points=5,
    rate_tolerance=.05, return_relative_tolerance=.05, time_fraction_tolerance=.05,
    min_success_rate=.8, max_collision_rate=.1, max_timeout_rate=.2,
    max_stagnation_rate=.2, stagnation_steps=50, stagnation_speed=.05)


def settings(value=None):
    c = copy.deepcopy(DEFAULTS if value is None else value)
    if set(c) != set(DEFAULTS) or c['schema'] != SCHEMA:
        raise ValueError('complete versioned monitoring settings required')
    for k in ('eval_every','tail_points','stagnation_steps'):
        if type(c[k]) is not int or c[k] < (3 if k=='tail_points' else 1):
            raise ValueError('invalid monitoring integer: '+k)
    for k in set(c)-{'schema','eval_every','tail_points','stagnation_steps'}:
        if isinstance(c[k],bool) or not math.isfinite(float(c[k])) or not 0<float(c[k])<=1:
            raise ValueError('invalid monitoring scalar: '+k)
    return c


def schedule(episodes, interval):
    if type(episodes) is not int or episodes<1 or type(interval) is not int or interval<1:
        raise ValueError('positive explicit episode/interval budgets required')
    return [0, *range(interval, episodes, interval), episodes]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def read_journal(path):
    """Read committed lines only; tolerate only the currently unfinished tail."""
    path=Path(path)
    if not path.exists():return [],dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest())
    data=path.read_bytes();end=data.rfind(b'\n')+1;prefix=data[:end]
    rows=[json.loads(line) for line in prefix.splitlines() if line.strip()]
    return rows,dict(bytes=end,sha256=hashlib.sha256(prefix).hexdigest(),unfinished_tail=bool(data[end:]))


def wilson(k,n):
    if type(n) is not int or n<1 or not 0<=k<=n:raise ValueError('valid binomial count required')
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half),min(1.,center+half)]


def episode_metrics(row,max_steps,spec):
    length=int(row['actual_length']);reason=row['termination_reason'];success=bool(row['success'])
    if not 1<=length<=max_steps or reason not in ('success','obstacle_collision','timeout'):
        raise ValueError('invalid complete strict episode')
    collision=bool(row.get('collision_records') or row.get('first_collision_agent_ids') or reason=='obstacle_collision')
    if collision!=(reason=='obstacle_collision') or success!=(reason=='success'):
        raise ValueError('collision/success cannot hide under another termination')
    trace=row.get('reach_avoid',{}).get('trace',[])
    active=uncertified=stalled=0;runs=[0]*4;longest=[0]*4
    for x in trace:
        i=int(x['agent_id'])
        if int(x['phase'])==3:runs[i]=0;continue
        active+=1
        emergency=str(x['action_mode']).startswith('UNCERTIFIED')
        uncertified+=emergency
        goal=np.asarray(x['selected_goal'],float)[10:13]
        far=float(np.linalg.norm(np.asarray(x['public_position'])-goal))>.8
        slow=float(np.linalg.norm(x['velocity']))<spec['stagnation_speed']
        stuck=bool((far and slow or emergency) and not x.get('local_goal_reached',False))
        stalled+=stuck;runs[i]=runs[i]+1 if stuck else 0;longest[i]=max(longest[i],runs[i])
    ids=row.get('first_collision_agent_ids',[])
    result=dict(actual_length=length,success=int(success),collision=int(collision),timeout=int(reason=='timeout'),
        found=int(bool(row.get('found',False))),termination_reason=reason,
        team_discounted_return=float(row['team_discounted_return']),
        team_undiscounted_return=float(row.get('team_undiscounted_return',0.)),
        penalized_completion_steps=length if success else int(max_steps),
        active_agent_steps=active,uncertified_agent_steps=uncertified,
        uncertified_fraction=uncertified/max(1,active),stalled_agent_steps=stalled,
        stalled_fraction=stalled/max(1,active),longest_stall_by_agent=longest,
        stagnation=int(max(longest)>=spec['stagnation_steps']),
        executor_search_collision=int(3 in ids and row.get('first_collision_phase')=='Search'),
        executor_other_phase_collision=int(3 in ids and row.get('first_collision_phase')!='Search'),
        searcher_collision=int(any(i in (0,1,2) for i in ids)))
    if not all(math.isfinite(result[k]) for k in ('team_discounted_return','team_undiscounted_return')):
        raise ValueError('nonfinite original task return')
    return result


def aggregate(rows,expected_count):
    if len(rows)!=expected_count or expected_count<1:
        raise ValueError('incomplete validation; never drop failed episodes')
    out=dict(episodes=expected_count)
    for key in ('success','collision','timeout','found','stagnation'):
        count=sum(int(r[key]) for r in rows)
        out[key+'_count']=count;out[key+'_rate']=count/expected_count
        out[key+'_ci95']=wilson(count,expected_count)
    for key in ('team_discounted_return','team_undiscounted_return','penalized_completion_steps','actual_length'):
        out[key]=float(np.mean([r[key] for r in rows]))
    denom=max(1,sum(r['active_agent_steps'] for r in rows))
    out['uncertified_fraction']=sum(r['uncertified_agent_steps'] for r in rows)/denom
    out['stalled_fraction']=sum(r['stalled_agent_steps'] for r in rows)/denom
    for key in ('executor_search_collision','executor_other_phase_collision','searcher_collision'):
        out[key+'_count']=sum(r[key] for r in rows)
    return out


def assess(curve,spec,max_steps):
    """Stability, task quality and learning gain are DIFFERENT reported facts."""
    base=dict(theoretical_convergence_proved=False,formal_performance_passed=False,
        statistical_convergence_proved=False,single_training_seed=True,
        interpretation='Predeclared empirical screen on development cases, not a theorem or final-test acceptance')
    n=spec['tail_points']
    if len(curve)<n+1:return dict(base,status='insufficient_evaluations',empirical_plateau=False,convergence_candidate=False)
    if any(curve[i]['episode']>=curve[i+1]['episode'] for i in range(len(curve)-1)):
        raise ValueError('evaluation points must be strictly increasing')
    if len({r['development_sha256'] for r in curve})!=1:
        raise ValueError('development set changed across checkpoints')
    first,last=curve[0],curve[-1];tail=curve[-n:]
    limits={k:spec['rate_tolerance'] for k in ('success_rate','collision_rate','timeout_rate','stagnation_rate','uncertified_fraction')}
    limits['team_discounted_return']=spec['return_relative_tolerance']*max(1.,abs(first['team_discounted_return']))
    limits['penalized_completion_steps']=spec['time_fraction_tolerance']*max_steps
    for row in curve:
        if any(not math.isfinite(float(row[k])) for k in limits):raise ValueError('nonfinite validation curve')
    spread={k:float(np.ptp([r[k] for r in tail])) for k in limits}
    stable=all(spread[k]<=limits[k]+1e-12 for k in limits)
    mean=lambda k:float(np.mean([r[k] for r in tail]))
    quality=dict(success=mean('success_rate')>=spec['min_success_rate'],
        collision=mean('collision_rate')<=spec['max_collision_rate'],
        timeout=mean('timeout_rate')<=spec['max_timeout_rate'],
        stagnation=mean('stagnation_rate')<=spec['max_stagnation_rate'])
    safety=mean('collision_rate')<=first['collision_rate']+spec['rate_tolerance']+1e-12
    gain=(mean('success_rate')-first['success_rate']>=1/first['episodes']-1e-12 or
          mean('success_rate')>=first['success_rate']-spec['rate_tolerance'] and
          (first['penalized_completion_steps']-mean('penalized_completion_steps')>limits['penalized_completion_steps'] or
           mean('team_discounted_return')-first['team_discounted_return']>limits['team_discounted_return']))
    learned=last['updates']>first['updates']
    status=('no_online_learning_updates' if not learned else 'still_changing' if not stable else
            'stable_but_ineffective' if not all(quality.values()) else 'stable_with_safety_regression' if not safety else
            'empirically_stable_candidate' if gain else 'stable_good_policy_without_clear_online_gain')
    return dict(base,status=status,empirical_plateau=stable,
        convergence_candidate=bool(learned and stable and all(quality.values()) and safety),
        task_quality_checks=quality,online_task_gain=bool(gain),collision_regression_guard=bool(safety),
        tail_episodes=[r['episode'] for r in tail],tail_ranges=spread,tolerances=limits,
        note='Episode0 is the learned policy AFTER explicit offline pretraining; gain here concerns subsequent online training')
