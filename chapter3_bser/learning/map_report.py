"""Paired native-map summaries; seeds and shared scenes remain separate units."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from chapter3_bser.learning.map_suite import load_plan,training_receipt,read
from chapter3_bser.learning.evaluation_records import verified_rows
from chapter3_bser.learning.checkpoint import digest,write_json
from chapter3_bser.learning.provenance import require_review

METRICS=('found_rate','success_rate','collision_rate','pre_found_collision_rate','found_penalized_steps','team_discounted_return','runtime_seconds','fallbacks')


def metric_values(rows):
    return dict(found_rate=np.asarray([r['found_step'] is not None for r in rows],float),
        success_rate=np.asarray([r['success'] for r in rows],float),
        collision_rate=np.asarray([r['termination_reason']=='obstacle_collision' for r in rows],float),
        pre_found_collision_rate=np.asarray([r['pre_found_collision'] for r in rows],float),
        found_penalized_steps=np.asarray([r['found_penalized_steps'] for r in rows],float),
        team_discounted_return=np.asarray([r['team_discounted_return'] for r in rows],float),
        runtime_seconds=np.asarray([r['map_metrics']['runtime_step_seconds'] for r in rows],float),
        fallbacks=np.asarray([r['map_metrics']['d2_fallbacks'] for r in rows],float))


def paired_interval(delta,*,draws,seed):
    """Crossed seed/scene bootstrap, sharing each scene draw across seeds."""
    data=np.asarray(delta,float)
    if data.ndim!=2 or not data.size or not np.all(np.isfinite(data)):raise ValueError('finite seed x scene paired deltas required')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        seeds=rng.integers(data.shape[0],size=data.shape[0]);scenes=rng.integers(data.shape[1],size=data.shape[1])
        values.append(float(data[np.ix_(seeds,scenes)].mean()))
    return dict(mean=float(data.mean()),lower=float(np.percentile(values,2.5)),upper=float(np.percentile(values,97.5)),
                training_replicates=data.shape[0],paired_scenes=data.shape[1],
                interval_scope='scene uncertainty only' if data.shape[0]==1 else 'crossed training-seed and held-out-scene uncertainty')


def csv_file(path,rows):
    if not rows:return
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(root,output,groups=None):
    root=Path(root).resolve();plan=load_plan(root);output=Path(output).resolve()
    selected=list(plan['spec']['groups'] if groups is None else groups)
    if not selected or len(selected)!=len(set(selected)) or set(selected)-set(plan['spec']['groups']):
        raise ValueError('unique prepared report groups required')
    if output.exists():raise FileExistsError(output)
    evaluations={};replicates=[];pairing=None;training_rows=[];diagnostics=[];validation_rows=[];diagnostic_sets=[]
    for job in plan['jobs']:
        if job['group'] not in selected:continue
        if job['seed'] is not None:trained=training_receipt(root,plan,job)
        else:trained=None
        if job['seed'] is not None and trained is None:raise ValueError('missing completed training: '+job['name'])
        rows,receipt=verified_rows(root/job['evaluation']/'episodes.jsonl',plan['source_review'])
        method='D2' if job['learner']=='ZERO' else f'D2_{job["learner"]}_repair_v2'
        if (receipt.get('map_condition')!=job['map_condition'] or receipt['manifest_sha256']!=digest(read(root/'inputs/evaluation.json'))
                or receipt['mechanism_only']!=plan['mechanism_only'] or receipt['method']!=method
                or receipt['completed']!=plan['spec']['evaluation_count']
                or trained is not None and receipt['policy_id']!=trained['model_state_sha256']):
            raise ValueError('map evaluation is not the native final policy in this plan')
        key=[tuple(r[k] for k in ('scenario_id','scenario_seed','scenario_sha256','environment_innovation_seed','max_steps','gamma',
                'task_protocol','reward_objective','source_reward_revision','waypoint_reward_timing')) for r in rows]
        if pairing is not None and key!=pairing:raise ValueError('map evaluations are not paired on identical physical scenarios')
        pairing=key
        expected_seed=job['seed']
        if any(r['training_seed']!=expected_seed or r['method']!=method or r['policy_id']!=receipt['policy_id'] for r in rows):raise ValueError('map evaluation learner/seed/policy mismatch')
        for r in rows:
            if r['found_penalized_steps']!=(r['max_steps'] if r['found_step'] is None else r['found_step']):raise ValueError('failure penalty metric changed')
        diagnostic_sets.append((job['group'],job['seed'],rows))
        metrics=metric_values(rows)
        if any(not np.all(np.isfinite(v)) for v in metrics.values()):raise ValueError('nonfinite map metrics')
        evaluations[(job['group'],job['seed'])]=metrics
        record=dict(group=job['group'],training_seed=job['seed'],scenarios=len(rows),policy_sha256=receipt['policy_id'],
            **{k:float(v.mean()) for k,v in metrics.items()})
        replicates.append(record)
        if trained is not None:
            if 'convergence' in plan['spec']:
                from chapter3_bser.learning.map_convergence import verified_diagnostic
                diagnosis=verified_diagnostic(root,plan,job);validation_rows.extend(diagnosis['curve'])
                diagnostics.append(dict(group=job['group'],training_seed=job['seed'],**diagnosis['assessment']))
            text=(root/job['train']/'episodes.jsonl').read_text(encoding='utf-8')
            for line in text.splitlines():
                r=json.loads(line)
                training_rows.append(dict(group=job['group'],training_seed=job['seed'],episode=r['episode'],environment_steps=r['global_step'],
                    team_discounted_return=r['team_discounted_return'],found=int(r['found_step'] is not None),success=int(r['success']),
                    collision=int(r['termination_reason']=='obstacle_collision')))
    groups=[]
    for group in selected:
        values=[r for r in replicates if r['group']==group]
        row=dict(group=group,training_replicates=0 if group.endswith('_ZERO') else len(values),policies=len(values),scenarios_per_policy=plan['spec']['evaluation_count'])
        if 'convergence' in plan['spec']:
            subset=[r for r in diagnostics if r['group']==group]
            row['all_seeds_provisional_plateau']=None if group.endswith('_ZERO') else all(r['provisional_plateau'] for r in subset)
        for key in METRICS:
            data=[r[key] for r in values];row[key]=statistics.mean(data)
            row[key+'_seed_sd']=statistics.stdev(data) if len(data)>1 else None
        groups.append(row)
    comparisons=[]
    for learner in ('ZERO','B2','B3'):
        old,new='OLD_'+learner,'NEW_'+learner
        if old not in selected or new not in selected:continue
        seeds=[None] if learner=='ZERO' else plan['spec']['seeds']
        for key in METRICS:
            delta=np.stack([evaluations[(new,s)][key]-evaluations[(old,s)][key] for s in seeds])
            ci=paired_interval(delta,draws=plan['spec']['bootstrap_draws'],seed=plan['spec']['statistics_seed'])
            if learner=='ZERO':ci.update(training_replicates=0,interval_scope='paired held-out-scene uncertainty; no learned policy')
            comparisons.append(dict(comparison=new+' minus '+old,metric=key,**ci))
    result=dict(schema='auv.active.map_summary.v1',source_review=plan['source_review'],plan_sha256=plan['sha256'],complete=True,
        selected_groups=selected,all_scheduled_groups_reported=set(selected)==set(plan['spec']['groups']),
        native_evaluation_only=True,mechanism_only=plan['mechanism_only'],groups=groups,comparisons=comparisons,
        bootstrap_draws=plan['spec']['bootstrap_draws'],performance_passed=False)
    require_review(plan['source_review']);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'summary.json',result);csv_file(output/'groups.csv',groups);csv_file(output/'per_seed.csv',replicates)
    from chapter3_bser.learning.map_diagnostics import write_tables
    write_tables(output,diagnostic_sets)
    csv_file(output/'paired_deltas.csv',comparisons);csv_file(output/'learning_curves.csv',training_rows)
    if diagnostics:
        write_json(output/'convergence.json',dict(per_seed=diagnostics,scope='Exploratory plateau screen; two seeds and 12 validation scenes do not prove convergence.'))
        csv_file(output/'validation_curves.csv',validation_rows)
    lines=['# OLD / NEW maps with independently adapted ZERO, B2 and B3','','All selected native policies completed; this does not assert a performance improvement.',
        'Selected groups: '+', '.join(selected)+'. All planned groups reported: '+str(result['all_scheduled_groups_reported'])+'.','',
        '| Group | Policies | Found | Success | Collision | Penalized found steps |','|---|---:|---:|---:|---:|---:|']
    for r in groups:lines.append(f'| {r["group"]} | {r["policies"]} | {r["found_rate"]:.4f} | {r["success_rate"]:.4f} | {r["collision_rate"]:.4f} | {r["found_penalized_steps"]:.2f} |')
    if diagnostics:
        lines+=['','## Fixed-validation convergence screen','','| Group | Seed | Status |','|---|---:|---|']
        for r in diagnostics:lines.append(f'| {r["group"]} | {r["training_seed"]} | {r["status"]} |')
        lines+=['','Inspect jobs/<job>/diagnostics/learning_curves.html for validation, training return and losses.',
                'A provisional plateau requires stable last three checkpoints, task gain over the untrained policy and no collision regression. It is not proof of convergence; report both seeds separately.']
    lines+=['','- Paired deltas are NEW minus OLD. Negative is better for collision, time and cost; positive is better for Found and Success.',
        '- Learning repeats are training seeds; shared evaluation scenes are a separate sampling axis.',
        '- One-seed intervals contain scene uncertainty only. ZERO has no training repeats.',
        '- Core planning timing covers successful proposals; runtime timing includes environment and actor. Report fallback frequency as well.',
        '- Native evaluation only: each model is evaluated with its own training map. No historical weights are restored or relabelled by preparation.',
        '- Source/config/checkpoint hashes and final-episode selection are retained. Outputs are never overwritten.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
