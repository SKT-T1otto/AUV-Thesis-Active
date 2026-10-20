"""Offline fixed-validation learning diagnostics; never select on final test scores."""
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch
from chapter3_bser.learning.checkpoint import digest,state_digest,write_json
from chapter3_bser.learning.evaluation_records import verified_rows
from chapter3_bser.learning.map_contract import metadata
from chapter3_bser.learning.provenance import require_review


def validate_settings(value):
    keys={'validation_count','generator_seed','tail_points','rate_tolerance','time_fraction_tolerance','return_relative_tolerance'}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('complete convergence settings required')
    c=copy.deepcopy(value)
    for key,minimum in (('validation_count',2),('generator_seed',0),('tail_points',3)):
        if type(c[key]) is not int or c[key]<minimum:raise ValueError('invalid convergence integer: '+key)
    if c['generator_seed']>=2**31:raise ValueError('invalid validation generator seed')
    for key in ('rate_tolerance','time_fraction_tolerance','return_relative_tolerance'):
        if type(c[key]) not in (int,float) or not np.isfinite(c[key]) or not 0<c[key]<=1:raise ValueError('invalid convergence tolerance')
    return c


def points(spec):
    if 'convergence' not in spec:return []
    return [0,*range(spec['checkpoint_interval'],spec['train_episodes'],spec['checkpoint_interval']),spec['train_episodes']]


def checkpoint_name(spec,episode):
    if episode==0:return None
    return 'checkpoint_final.pt' if episode==spec['train_episodes'] else f'checkpoint_episode_{episode:06d}.pt'


def checkpoint_files(spec):
    return [checkpoint_name(spec,e) for e in points(spec) if e]


def assess(curve,settings,max_steps,*,learned=True):
    """Predeclared practical plateau screen, not a statistical convergence proof."""
    n=settings['tail_points']
    if len(curve)<n+1:return dict(status='insufficient_checkpoints',provisional_plateau=False)
    if any(not np.isfinite(r[k]) for r in curve for k in ('success_rate','found_rate','collision_rate','found_penalized_steps','team_discounted_return')):
        raise ValueError('nonfinite validation curve')
    tail=curve[-n:];initial=curve[0];last=curve[-1]
    limits={k:settings['rate_tolerance'] for k in ('success_rate','found_rate','collision_rate')}
    limits.update(found_penalized_steps=settings['time_fraction_tolerance']*max_steps,
                  team_discounted_return=settings['return_relative_tolerance']*max(abs(initial['team_discounted_return']),1.))
    spread={k:float(np.ptp([r[k] for r in tail])) for k in limits}
    stable=all(spread[k]<=limits[k]+1e-12 for k in limits)
    mean=lambda k:float(np.mean([r[k] for r in tail]))
    task_gain=(mean('success_rate')-initial['success_rate']>=1/settings['validation_count']-1e-12
        or mean('found_rate')-initial['found_rate']>=1/settings['validation_count']-1e-12
        or initial['found_penalized_steps']-mean('found_penalized_steps')>=limits['found_penalized_steps'])
    safe=mean('collision_rate')<=initial['collision_rate']+settings['rate_tolerance']+1e-12
    status=('no_learning_updates' if not learned else 'still_changing' if not stable else
            'stable_without_clear_task_gain' if not task_gain else 'stable_with_collision_regression' if not safe else 'provisional_plateau')
    return dict(status=status,provisional_plateau=status=='provisional_plateau',tail_episodes=[r['episode'] for r in tail],
                tail_range=spread,tolerances=limits,task_gain_over_initial=bool(task_gain),collision_guard=bool(safe),
                final_minus_initial={k:last[k]-initial[k] for k in limits},
                interpretation='Practical screen on a small fixed validation set; two seeds cannot establish general convergence.')


def evaluate_point(root,plan,job,episode,trained):
    """Only called by explicit Linux research. Automatic checks never restore files."""
    from chapter3_bser.learning.map_suite import read
    from chapter3_bser.learning.evaluate import _evaluate_loaded,preserved_rng
    from chapter3_bser.learning.train import load_for_evaluation
    from chapter3_bser.learning.model import build_model
    config=read(root/job['config']);checkpoint=checkpoint_name(plan['spec'],episode)
    output=root/'jobs'/job['name']/'diagnostics'/f'episode_{episode:06d}'
    with preserved_rng():
        if episode:
            model,loaded,payload=load_for_evaluation(root/job['train']/checkpoint)
            if loaded!=config or payload['episodes']!=episode or payload.get('initial_model_state_sha256')!=trained['initial_model_state_sha256']:
                raise ValueError('diagnostic checkpoint episode/config/initial weights mismatch')
            if episode==plan['spec']['train_episodes'] and payload['model_state_sha256']!=trained['model_state_sha256']:
                raise ValueError('diagnostic final policy differs from completed training')
        else:
            random.seed(config['seed']);np.random.seed(config['seed']%2**32);torch.manual_seed(config['seed'])
            model=build_model(config);model.prep_rollouts(device='cpu')
            state=state_digest(model.training_state_dict())
            if state!=trained['initial_model_state_sha256']:raise ValueError('reconstructed initial weights differ from training')
            payload=dict(model_state_sha256=state,training_scenario_ids=[],training_scenario_seeds=[],training_scenario_fingerprints=[],**metadata(config))
        before=state_digest(model.training_state_dict())
        _evaluate_loaded(model,config,payload,root/'inputs/validation.json',output,identity=plan['source_review'],
                         checkpoint=None if checkpoint is None else root/job['train']/checkpoint)
        if state_digest(model.training_state_dict())!=before:raise ValueError('diagnostic evaluation mutated frozen policy')
    return before


def point_rows(root,plan,job,episode,trained,*,allow_execute=True):
    from chapter3_bser.learning.map_suite import read,sha
    directory=root/'jobs'/job['name']/'diagnostics'/f'episode_{episode:06d}'
    binding=dict(schema='auv.active.map_diagnostic_point.v1',source_review=plan['source_review'],plan_sha256=plan['sha256'],
        job=job['name'],episode=episode,training_receipt_sha256=sha(root/'jobs'/job['name']/'training_receipt.json'),
        checkpoint_sha256=None if episode==0 else trained['diagnostic_checkpoints'][checkpoint_name(plan['spec'],episode)])
    marker=directory/'diagnostic_identity.json'
    if marker.exists():
        existing=read(marker)
        if any(existing.get(k)!=v for k,v in binding.items()):raise ValueError('diagnostic checkpoint binding changed')
        policy=existing['policy_sha256']
    else:
        if not allow_execute:raise ValueError('missing diagnostic point; run diagnose first')
        if directory.exists():raise FileExistsError('incomplete diagnostic output retained: '+str(directory))
        policy=evaluate_point(root,plan,job,episode,trained)
    rows,receipt=verified_rows(directory/'episodes.jsonl',plan['source_review'])
    expected=read(root/'inputs/validation.json')
    config=read(root/job['config'])
    if (receipt.get('map_condition')!=job['map_condition'] or receipt['policy_id']!=policy or receipt['method']!=config['method']
            or receipt['manifest_sha256']!=digest(expected) or receipt['completed']!=plan['spec']['convergence']['validation_count']
            or receipt['mechanism_only']!=plan['mechanism_only']
            or [(r['scenario_id'],r['scenario_seed'],r['scenario_sha256']) for r in rows]!=[(s['scenario_id'],s['scenario_seed'],digest(s)) for s in expected['scenarios']]
            or any(r['training_seed']!=job['seed'] or r['policy_id']!=policy or r['method']!=config['method'] for r in rows)):
        raise ValueError('diagnostic evaluation is not the paired fixed validation policy')
    if episode==0 and policy!=trained['initial_model_state_sha256']:raise ValueError('initial diagnostic policy changed')
    if episode==plan['spec']['train_episodes'] and policy!=trained['model_state_sha256']:raise ValueError('final diagnostic policy changed')
    if not marker.exists():write_json(marker,dict(**binding,policy_sha256=policy))
    return rows,policy


def verified_diagnostic(root,plan,job):
    from chapter3_bser.learning.map_suite import read,sha,training_receipt
    from chapter3_bser.learning.map_report import metric_values
    trained=training_receipt(root,plan,job)
    if trained is None:raise ValueError('missing completed diagnostic training')
    path=root/'jobs'/job['name']/'diagnostics'/'convergence.json'
    result=read(path);curve=[];bindings={}
    for episode in points(plan['spec']):
        rows,policy=point_rows(root,plan,job,episode,trained,allow_execute=False)
        metrics=metric_values(rows)
        curve.append(dict(group=job['group'],training_seed=job['seed'],episode=episode,scenarios=len(rows),policy_sha256=policy,
                          **{k:float(v.mean()) for k,v in metrics.items()}))
        marker=path.parent/f'episode_{episode:06d}'/'diagnostic_identity.json'
        bindings[str(episode)]=dict(marker_sha256=sha(marker),evaluation_receipt_sha256=sha(marker.parent/'evaluation_identity.json'))
    losses=loss_rows(root,job);finite=all(np.isfinite(r[k]) for r in losses for k in ('critic_loss','actor_loss') if r[k] is not None)
    if (result.get('source_review')!=plan['source_review'] or result.get('plan_sha256')!=plan['sha256'] or result.get('job')!=job['name']
            or result.get('map_condition')!=job['map_condition'] or result.get('curve')!=curve or result.get('point_bindings')!=bindings
            or result.get('training_receipt_sha256')!=sha(root/'jobs'/job['name']/'training_receipt.json')
            or result.get('assessment')!=assess(curve,plan['spec']['convergence'],plan['spec']['max_steps'],learned=bool(losses) and finite)):
        raise ValueError('diagnostic summary/source/curve changed')
    return result


def diagnose(root,plan,jobs):
    from chapter3_bser.learning.map_suite import training_receipt,read,sha
    from chapter3_bser.learning.map_report import metric_values,csv_file
    if 'convergence' not in plan['spec']:raise ValueError('this plan has no fixed validation diagnostics')
    result=[]
    for job in jobs:
        if job['seed'] is None:continue
        trained=training_receipt(root,plan,job)
        if trained is None:raise ValueError('completed training required before offline diagnostics')
        curve=[];bindings={}
        for episode in points(plan['spec']):
            rows,policy=point_rows(root,plan,job,episode,trained)
            metrics=metric_values(rows)
            curve.append(dict(group=job['group'],training_seed=job['seed'],episode=episode,scenarios=len(rows),policy_sha256=policy,
                              **{k:float(v.mean()) for k,v in metrics.items()}))
            marker=root/'jobs'/job['name']/'diagnostics'/f'episode_{episode:06d}'/'diagnostic_identity.json'
            bindings[str(episode)]=dict(marker_sha256=sha(marker),evaluation_receipt_sha256=sha(marker.parent/'evaluation_identity.json'))
            print(f'[validation] {job["name"]} episode={episode}',flush=True)
        losses=loss_rows(root,job)
        finite_losses=all(np.isfinite(r[k]) for r in losses for k in ('critic_loss','actor_loss') if r[k] is not None)
        outcome=assess(curve,plan['spec']['convergence'],plan['spec']['max_steps'],learned=bool(losses) and finite_losses)
        value=dict(schema='auv.active.map_convergence.v1',source_review=plan['source_review'],plan_sha256=plan['sha256'],
                   job=job['name'],map_condition=job['map_condition'],curve=curve,point_bindings=bindings,assessment=outcome,
                   training_receipt_sha256=sha(root/'jobs'/job['name']/'training_receipt.json'),losses_finite=finite_losses,
                   selection='final completed episode; validation diagnoses only; final test never selects a checkpoint',performance_passed=False)
        path=root/'jobs'/job['name']/'diagnostics'/'convergence.json'
        if path.exists():
            if read(path)!=value:raise ValueError('diagnostic summary changed')
        else:
            require_review(plan['source_review']);write_json(path,value)
            csv_file(path.with_name('validation_curve.csv'),curve)
            csv_file(path.with_name('optimizer_curve.csv'),losses)
            write_dashboard(path.with_name('learning_curves.html'),curve,training_rows(root,job),losses)
        result.append(dict(job=job['name'],stage='diagnose',status='completed',assessment=outcome['status']))
    return result


def training_rows(root,job):
    return [json.loads(line) for line in (root/job['train']/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


def loss_rows(root,job):
    rows=[]
    for line in (root/job['train']/'training_metrics.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        item=json.loads(line);agents=item.get('agents',[])
        if not agents:continue
        critic=[a['critic_loss'] for a in agents];actor=[a['actor_loss'] for a in agents if a.get('actor_updated') and a.get('actor_loss') is not None]
        rows.append(dict(global_step=item['global_step'],update=item['update'],critic_loss=float(np.mean(critic)),
                         actor_loss=None if not actor else float(np.mean(actor))))
    return rows


def write_dashboard(path,curve,training,losses):
    """Self-contained browser charts without adding a runtime dependency."""
    data=json.dumps(dict(validation=curve,training=training,losses=losses),allow_nan=False).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>RL learning diagnostics</title>
<style>body{font:15px system-ui;max-width:1100px;margin:30px auto;color:#203040}svg{width:100%;background:#fafcfe;border:1px solid #ddd}section{margin:24px 0}h2{font-size:18px}p{line-height:1.6}</style>
<h1>RL learning diagnostics</h1><p>Validation uses the same held-out scenes at every checkpoint, exploration off. Training return uses a trailing 20-episode mean. Losses are diagnostics, not convergence criteria. Hover on points for values. A plateau alone does not establish effective learning.</p><main></main>
<script>const data=__DATA__;
function plot(title,rows,xkey,ykey,windowSize=1){let pts=rows.filter(r=>r[ykey]!==null&&Number.isFinite(r[ykey])).map((r,i,a)=>[r[xkey],a.slice(Math.max(0,i-windowSize+1),i+1).reduce((v,z)=>v+z[ykey],0)/Math.min(i+1,windowSize)]);if(!pts.length)return;
let xmin=Math.min(...pts.map(p=>p[0])),xmax=Math.max(...pts.map(p=>p[0])),ymin=Math.min(...pts.map(p=>p[1])),ymax=Math.max(...pts.map(p=>p[1]));if(xmax===xmin)xmax++;if(ymax===ymin){ymax+=.5;ymin-=.5;}
const X=x=>75+(x-xmin)/(xmax-xmin)*900,Y=y=>225-(y-ymin)/(ymax-ymin)*185;let svg='<svg viewBox="0 0 1020 280" role="img"><title>'+title+'</title>';
for(let i=0;i<=4;i++){let v=ymin+(ymax-ymin)*i/4;svg+=`<line x1="75" x2="975" y1="${Y(v)}" y2="${Y(v)}" stroke="#ddd"/><text x="68" y="${Y(v)+5}" text-anchor="end" font-size="12">${v.toPrecision(4)}</text>`;let x=xmin+(xmax-xmin)*i/4;svg+=`<text x="${X(x)}" y="245" text-anchor="middle" font-size="12">${Math.round(x)}</text>`;}
svg+='<polyline fill="none" stroke="#1769aa" stroke-width="2" points="'+pts.map(p=>`${X(p[0])},${Y(p[1])}`).join(' ')+'"/>';if(pts.length<100)for(const p of pts)svg+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="#1769aa"><title>${xkey}=${p[0]}, ${ykey}=${p[1]}</title></circle>`;
svg+=`<text x="525" y="270" text-anchor="middle" font-size="12">${xkey}</text></svg>`;let s=document.createElement('section'),h=document.createElement('h2');h.textContent=title;s.append(h);s.insertAdjacentHTML('beforeend',svg);document.querySelector('main').append(s);}
for(const k of ['success_rate','found_rate','collision_rate','found_penalized_steps','team_discounted_return','fallbacks'])plot('Fixed validation: '+k,data.validation,'episode',k);
plot('Training return: trailing 20 episodes',data.training,'episode','team_discounted_return',20);
for(const k of ['actor_loss','critic_loss'])plot(k+': trailing 100 updates',data.losses,'global_step',k,100);
</script>'''.replace('__DATA__',data)
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(html)
