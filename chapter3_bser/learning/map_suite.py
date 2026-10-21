"""Native OLD/NEW x ZERO/B2/B3 experiments with independent fresh training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from project_paths import ROOT
from chapter3_bser.learning.checkpoint import digest,write_json,state_digest
from chapter3_bser.learning.config import validate,scenario_fingerprint
from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review
from chapter3_bser.learning.scenarios import export,validate_manifest
from chapter3_bser.learning.suite import run_lock
from chapter3_bser.learning.map_contract import identity as map_identity,SCHEMA as MAP_SCHEMA

GROUPS=('OLD_ZERO','NEW_ZERO','OLD_B2','NEW_B2','OLD_B3','NEW_B3')
SPEC_SCHEMA='auv.active.map_experiment_spec.v1'
PLAN_SCHEMA='auv.active.map_suite.v1'
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_spec(spec,*,mechanism=False):
    c=copy.deepcopy(spec)
    keys={'schema','groups','seeds','train_episodes','max_steps','evaluation_count','evaluation_generator_seed',
          'training_generator_seed_base','checkpoint_interval','new_map_config','bootstrap_draws','statistics_seed'}
    if not isinstance(c,dict) or set(c)-{'convergence'}!=keys or c['schema']!=SPEC_SCHEMA:raise ValueError('complete map experiment specification required')
    if (not isinstance(c['groups'],list) or not c['groups'] or len(set(c['groups']))!=len(c['groups']) or set(c['groups'])-set(GROUPS)
            or not isinstance(c['seeds'],list) or not c['seeds'] or len(set(c['seeds']))!=len(c['seeds'])
            or any(type(s) is not int or not 0<=s<2**31 for s in c['seeds'])):raise ValueError('unique supported groups and training seeds required')
    for name in ('train_episodes','max_steps','evaluation_count','checkpoint_interval','bootstrap_draws'):
        if type(c[name]) is not int or c[name]<1:raise ValueError('positive experiment budget required: '+name)
    for name in ('evaluation_generator_seed','training_generator_seed_base','statistics_seed'):
        if type(c[name]) is not int or not 0<=c[name]<2**31:raise ValueError('invalid experiment seed')
    if c['training_generator_seed_base']+max(c['seeds'])>=2**31:raise ValueError('training generator seed overflow')
    if mechanism and (c['max_steps']>32 or c['train_episodes']>8 or c['evaluation_count']>8):raise ValueError('mechanism budget exceeded')
    from chapter3_bser.experiments.coupled_search_v1.core_config import validate_core_config
    c['new_map_config']=validate_core_config(c['new_map_config'])
    if c['new_map_config']['enabled'] is not True:raise ValueError('experiment NEW map must be explicitly enabled')
    if c['max_steps']>c['new_map_config']['history_steps']:raise ValueError('complete public evidence history must cover the episode horizon')
    if 'convergence' in c:
        from chapter3_bser.learning.map_convergence import validate_settings,points
        c['convergence']=validate_settings(c['convergence'])
        if len(points(c))<c['convergence']['tail_points']+1:raise ValueError('convergence needs an initial policy and enough saved tail checkpoints')
    return c


def jobs_for(spec):
    return [dict(name=group if group.endswith('_ZERO') else f'{group}_seed{seed}',group=group,map_id=group.split('_')[0],
                 learner=group.split('_')[1],seed=seed) for group in spec['groups'] for seed in ([None] if group.endswith('_ZERO') else spec['seeds'])]


def disjoint(training,evaluation):
    for key in (lambda s:s['scenario_id'],lambda s:s['scenario_seed'],scenario_fingerprint):
        if {key(s) for s in training}&{key(s) for s in evaluation}:raise ValueError('training/evaluation physical tasks, IDs or seeds overlap')


def namespace_manifest(manifest,namespace):
    """Preserve physical inputs and original generated IDs; prevent seed-ID aliasing."""
    value=copy.deepcopy(manifest);value['suite_namespace']=namespace
    for scene in value['scenarios']:
        scene['source_scenario_id']=scene['scenario_id']
        scene['scenario_id']=namespace+'__'+scene['source_scenario_id']
    return value


def _config_for(root,job,spec,mechanism):
    method='B2' if job['learner']=='ZERO' else job['learner']
    c=read(ROOT/f'configs/{method.lower()}{"_mechanism" if mechanism else ""}.json')
    c.update(max_steps=spec['max_steps'],total_main_trajectories=spec['train_episodes'],checkpoint_interval=spec['checkpoint_interval'],
        map_experiment=dict(schema=MAP_SCHEMA,map_id=job['map_id'],research_config=None if job['map_id']=='OLD' else spec['new_map_config'],
                            protocol_sha256=digest(dict(spec=spec,mechanism_only=bool(mechanism),protocol='native_maps_fresh_paired_order_v1'))),
        output_dir=str(root/f'jobs/{job["name"]}/train'),max_total_environment_steps=min(256,spec['max_steps']*spec['train_episodes']) if mechanism else None)
    if mechanism:c['rl'].update(batch_size=4,warmup_steps=max(4,spec['max_steps']//2),update_frequency=max(4,spec['max_steps']//2))
    if job['seed'] is not None:
        path=root/f'inputs/train_seed{job["seed"]}.json'
        c.update(seed=job['seed'],scenario_manifest=str(path),scenario_manifest_sha256=sha(path),scenario_schedule=list(range(spec['train_episodes'])))
    return validate(c)


def prepare(root,spec,*,mechanism=False,training_manifests=None,evaluation_manifest=None,exclude_runs=None,validation_manifest=None):
    source=verify_reviewed_changes();spec=validate_spec(spec,mechanism=mechanism)
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    if 'collision_terminal' not in root.parts:raise ValueError('map suite root must contain collision_terminal')
    protocol=digest(dict(spec=spec,mechanism_only=bool(mechanism),protocol='native_maps_fresh_paired_order_v1'))
    inputs=root/'inputs';inputs.mkdir(parents=True,exist_ok=False)
    validation=dict(profile='M20_MOVING_UNKNOWN_MULTI',max_steps=spec['max_steps'])
    if evaluation_manifest is None:
        raw=inputs/'generated_evaluation.json'
        export(raw,count=spec['evaluation_count'],seed=spec['evaluation_generator_seed'],max_steps=spec['max_steps'])
        write_json(inputs/'evaluation.json',namespace_manifest(read(raw),f'eval_gen{spec["evaluation_generator_seed"]}'))
    else:write_json(inputs/'evaluation.json',evaluation_manifest)
    evaluation=read(inputs/'evaluation.json')
    eval_scenes=validate_manifest(evaluation,validation,evaluation=True,mechanism=mechanism)
    if len(eval_scenes)!=spec['evaluation_count']:raise ValueError('evaluation count differs from specification')
    validation_scenes=[]
    if 'convergence' in spec:
        if validation_manifest is None:
            raw=inputs/'generated_validation.json';settings=spec['convergence']
            export(raw,count=settings['validation_count'],seed=settings['generator_seed'],max_steps=spec['max_steps'])
            validation_manifest=namespace_manifest(read(raw),f'validation_gen{settings["generator_seed"]}')
        write_json(inputs/'validation.json',validation_manifest)
        validation_scenes=validate_manifest(validation_manifest,validation,evaluation=True,mechanism=mechanism)
        if len(validation_scenes)!=spec['convergence']['validation_count']:raise ValueError('validation count differs from specification')
        disjoint(validation_scenes,eval_scenes)
    exclusions=[]
    for excluded_root in exclude_runs or ():
        excluded_root=Path(excluded_root).resolve();previous=load_plan(excluded_root)
        names=['inputs/evaluation.json',*(f'inputs/train_seed{s}.json' for s in previous['spec']['seeds'])]
        if 'convergence' in previous['spec']:names.append('inputs/validation.json')
        for name in names:
            manifest=read(excluded_root/name)
            disjoint(manifest['scenarios'],eval_scenes)
            if validation_scenes:disjoint(manifest['scenarios'],validation_scenes)
            destination=f'excluded_{len(exclusions):03d}.json'
            with (inputs/destination).open('xb') as stream:stream.write((excluded_root/name).read_bytes())
            exclusions.append(dict(asset='inputs/'+destination,source_plan_sha256=previous['sha256'],source_manifest_sha256=previous['assets'][name]))
    for seed in spec['seeds']:
        path=inputs/f'train_seed{seed}.json'
        if training_manifests is None:
            generator_seed=spec['training_generator_seed_base']+seed;raw=inputs/f'generated_train_seed{seed}.json'
            export(raw,split='train',count=spec['train_episodes'],seed=generator_seed,max_steps=spec['max_steps'])
            write_json(path,namespace_manifest(read(raw),f'train_gen{generator_seed}'))
        else:write_json(path,training_manifests[seed])
        train=validate_manifest(read(path),validation,evaluation=False,mechanism=mechanism)
        if len(train)!=spec['train_episodes']:raise ValueError('one frozen training task per scheduled episode required')
        disjoint(train,eval_scenes)
        if validation_scenes:disjoint(train,validation_scenes)
    jobs=jobs_for(spec)
    for job in jobs:
        c=_config_for(root,job,spec,mechanism)
        job.update(config=f'inputs/{job["name"]}.json',train=f'jobs/{job["name"]}/train',evaluation=f'jobs/{job["name"]}/evaluation',map_condition=map_identity(c))
        write_json(root/job['config'],c)
    write_json(inputs/'spec.json',spec)
    assets={p.relative_to(root).as_posix():sha(p) for p in inputs.iterdir() if p.is_file()}
    plan=dict(schema=PLAN_SCHEMA,source_review=source,protocol_sha256=protocol,mechanism_only=bool(mechanism),spec=spec,
        jobs=jobs,assets=assets,excluded_manifests=exclusions,training_protocol='fresh weights; per-seed manifest order; innovation seed=training_seed+episode',
        checkpoint_selection='last fully completed episode',native_evaluation_only=True,performance_passed=False)
    plan['sha256']=digest(plan);require_review(source);write_json(root/'plan.json',plan)
    load_plan(root)
    from chapter3_bser.learning.map_convergence import points
    return dict(executed=True,root=str(root),plan_sha256=plan['sha256'],training_jobs=sum(j['seed'] is not None for j in jobs),evaluation_jobs=len(jobs),
        diagnostic_episodes=sum(j['seed'] is not None for j in jobs)*len(points(spec))*spec.get('convergence',{}).get('validation_count',0),
        max_training_steps=sum(j['seed'] is not None for j in jobs)*spec['train_episodes']*spec['max_steps'],
        max_evaluation_steps=len(jobs)*spec['evaluation_count']*spec['max_steps'],performance_passed=False)


def load_plan(root):
    root=Path(root).resolve();plan=read(root/'plan.json')
    if plan.get('schema')!=PLAN_SCHEMA or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'}):raise ValueError('map suite plan identity mismatch')
    require_review(plan['source_review']);spec=validate_spec(plan['spec'],mechanism=plan['mechanism_only'])
    protocol=digest(dict(spec=spec,mechanism_only=plan['mechanism_only'],protocol='native_maps_fresh_paired_order_v1'))
    if protocol!=plan['protocol_sha256'] or not plan.get('native_evaluation_only'):raise ValueError('map experiment protocol mismatch')
    expected=jobs_for(spec)
    if len(expected)!=len(plan['jobs']):raise ValueError('map suite job inventory mismatch')
    actual_assets={p.relative_to(root).as_posix():sha(p) for p in (root/'inputs').rglob('*') if p.is_file()}
    if actual_assets!=plan['assets']:raise ValueError('map suite prepared input changed')
    configs=[]
    eval_scenes=validate_manifest(read(root/'inputs/evaluation.json'),dict(profile='M20_MOVING_UNKNOWN_MULTI',max_steps=spec['max_steps']),evaluation=True,mechanism=plan['mechanism_only'])
    if len(eval_scenes)!=spec['evaluation_count']:raise ValueError('evaluation count changed')
    validation_scenes=[]
    if 'convergence' in spec:
        validation_scenes=validate_manifest(read(root/'inputs/validation.json'),dict(profile='M20_MOVING_UNKNOWN_MULTI',max_steps=spec['max_steps']),evaluation=True,mechanism=plan['mechanism_only'])
        if len(validation_scenes)!=spec['convergence']['validation_count']:raise ValueError('validation count changed')
        disjoint(validation_scenes,eval_scenes)
    for item in plan.get('excluded_manifests',[]):
        if item['asset'] not in plan['assets'] or item['source_manifest_sha256']!=plan['assets'][item['asset']]:raise ValueError('excluded development manifest changed')
        disjoint(read(root/item['asset'])['scenarios'],eval_scenes)
        if validation_scenes:disjoint(read(root/item['asset'])['scenarios'],validation_scenes)
    for base,job in zip(expected,plan['jobs']):
        if any(job.get(k)!=v for k,v in base.items()):raise ValueError('map suite group/seed/job mismatch')
        for key,expected_path in (('config',f'inputs/{job["name"]}.json'),('train',f'jobs/{job["name"]}/train'),('evaluation',f'jobs/{job["name"]}/evaluation')):
            if job[key]!=expected_path:raise ValueError('unsafe or mismatched job path')
        c=validate(read(root/job['config']));condition=map_identity(c)
        if c!=_config_for(root,base,spec,plan['mechanism_only']):raise ValueError('job configuration differs from preserved learner contract and frozen budgets')
        if (condition!=job['map_condition'] or condition['map_id']!=job['map_id'] or condition['protocol_sha256']!=protocol
                or c['map_experiment']['research_config']!=(None if job['map_id']=='OLD' else spec['new_map_config'])
                or c['max_steps']!=spec['max_steps'] or c['total_main_trajectories']!=spec['train_episodes']
                or c['output_dir']!=str(root/job['train']) or c['checkpoint_interval']!=spec['checkpoint_interval']
                or c['baseline']!=f'D2_{"B2" if job["learner"]=="ZERO" else job["learner"]}_repair_v2'):
            raise ValueError('job map, learning method or budget differs from frozen plan')
        if job['seed'] is not None:
            path=root/f'inputs/train_seed{job["seed"]}.json'
            if (c['seed']!=job['seed'] or c.get('scenario_manifest')!=str(path) or c.get('scenario_manifest_sha256')!=sha(path)
                    or c.get('scenario_schedule')!=list(range(spec['train_episodes']))):raise ValueError('paired training schedule changed')
            scenes=validate_manifest(read(path),c,evaluation=False,mechanism=plan['mechanism_only']);disjoint(scenes,eval_scenes)
            if validation_scenes:disjoint(scenes,validation_scenes)
        configs.append(c)
    # Apart from map identity/output location, a paired learner gets identical
    # training configuration, including initial seed and explicit scene order.
    for learner in ('B2','B3'):
        for seed in spec['seeds']:
            pair=[c for c,j in zip(configs,plan['jobs']) if j['learner']==learner and j['seed']==seed]
            if len(pair)==2:
                normalized=[{k:v for k,v in c.items() if k not in ('map_experiment','output_dir')} for c in pair]
                if normalized[0]!=normalized[1]:raise ValueError('paired learner hyperparameters differ')
    return plan


def training_receipt(root,plan,job):
    root=Path(root);path=root/'jobs'/job['name']/'training_receipt.json'
    if not path.exists():return None
    row=read(path);summary=read(root/job['train']/'summary.json')
    if (row.get('source_review')!=plan['source_review'] or row.get('plan_sha256')!=plan['sha256']
            or row.get('config_sha256')!=plan['assets'][job['config']] or row.get('map_condition')!=job['map_condition']
            or row.get('checkpoint_sha256')!=sha(root/job['train']/'checkpoint_final.pt')
            or row.get('summary_sha256')!=sha(root/job['train']/'summary.json')
            or row.get('episodes_sha256')!=sha(root/job['train']/'episodes.jsonl')
            or row.get('metrics_sha256')!=sha(root/job['train']/'training_metrics.jsonl')
            or summary.get('map_condition')!=job['map_condition'] or summary.get('episodes')!=plan['spec']['train_episodes']
            or summary.get('optimizer_updates',0)<1 or not any(summary.get('actor_optimizer_steps',[]))):
        raise ValueError('incomplete, unlearned or changed map training receipt: '+job['name'])
    if 'convergence' in plan['spec']:
        from chapter3_bser.learning.map_convergence import checkpoint_files
        expected={name:sha(root/job['train']/name) for name in checkpoint_files(plan['spec'])}
        if row.get('diagnostic_checkpoints')!=expected:raise ValueError('diagnostic checkpoint inventory changed')
    return row


def _record_training(root,plan,job,trainer,summary):
    if summary['episodes']!=plan['spec']['train_episodes'] or summary['optimizer_updates']<1 or not any(summary['actor_optimizer_steps']):
        raise ValueError('scheduled training did not complete or model never learned')
    row=dict(source_review=plan['source_review'],plan_sha256=plan['sha256'],config_sha256=plan['assets'][job['config']],
        map_condition=job['map_condition'],checkpoint_sha256=sha(root/job['train']/'checkpoint_final.pt'),
        summary_sha256=sha(root/job['train']/'summary.json'),initial_model_state_sha256=trainer.initial_model_state_sha256,
        episodes_sha256=sha(root/job['train']/'episodes.jsonl'),metrics_sha256=sha(root/job['train']/'training_metrics.jsonl'),
        model_state_sha256=state_digest(trainer.model.training_state_dict()))
    if 'convergence' in plan['spec']:
        from chapter3_bser.learning.map_convergence import checkpoint_files
        row['diagnostic_checkpoints']={name:sha(root/job['train']/name) for name in checkpoint_files(plan['spec'])}
    other=next((j for j in plan['jobs'] if j['learner']==job['learner'] and j['seed']==job['seed'] and j['map_id']!=job['map_id']),None)
    if other is not None:
        existing=training_receipt(root,plan,other)
        if existing is not None and existing['initial_model_state_sha256']!=row['initial_model_state_sha256']:raise ValueError('paired initial weights differ')
    path=root/'jobs'/job['name']/'training_receipt.json'
    if path.exists():raise FileExistsError(path)
    require_review(plan['source_review']);write_json(path,row)
    return row


def require_linux_validation(path,source):
    if platform.system()!='Linux':raise ValueError('formal map suite execution requires an actual Linux host')
    if path is None:raise ValueError('--validation from current Linux checkout verification is required')
    value=read(path)
    if (not value.get('passed') or value.get('linux_execution') is not True or value.get('source')!=source
            or not value.get('checks',{}).get('passed') or not value.get('map_workflow_verified')):
        raise ValueError('current Linux map-workflow verification missing or failed')
    head=subprocess.run(['git','-c','core.fsmonitor=false','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','-c','core.fsmonitor=false','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    if (value.get('git',{}).get('head')!=head or dirty
            or value.get('package_after',{}).get('package_sha256')!=sha(ROOT/'PACKAGE_SHA256.json')):
        raise ValueError('Linux validation is for another commit/package or checkout is dirty')


def run(root,*,stage='all',groups=None,seeds=None,validation=None):
    root=Path(root).resolve();plan=load_plan(root)
    if stage not in ('train','diagnose','evaluate','all'):raise ValueError('invalid map suite stage')
    selected_groups=set(groups or plan['spec']['groups']);selected_seeds=set(seeds or plan['spec']['seeds'])
    if selected_groups-set(plan['spec']['groups']) or selected_seeds-set(plan['spec']['seeds']):raise ValueError('requested group/seed absent from prepared map suite')
    selected=[j for j in plan['jobs'] if j['group'] in selected_groups and (j['seed'] is None or j['seed'] in selected_seeds)]
    if not plan['mechanism_only']:require_linux_validation(validation,plan['source_review'])
    else:
        if 'convergence' in plan['spec']:raise ValueError('checkpoint convergence evaluation is manual Linux research, not automatic mechanism verification')
        episodes=sum(plan['spec']['train_episodes'] for j in selected if j['seed'] is not None) if stage!='evaluate' else 0
        episodes+=len(selected)*plan['spec']['evaluation_count'] if stage!='train' else 0
        if episodes>8 or episodes*plan['spec']['max_steps']>256:raise ValueError('selected mechanism execution exceeds eight episodes x32 steps')
    results=[]
    with run_lock(root):
        for phase in ((('train','diagnose','evaluate') if 'convergence' in plan['spec'] else ('train','evaluate')) if stage=='all' else (stage,)):
            if phase=='diagnose':
                from chapter3_bser.learning.map_convergence import diagnose
                results.extend(diagnose(root,plan,selected));continue
            for job in selected:
                load_plan(root);config=read(root/job['config'])
                if phase=='train':
                    if job['seed'] is None:continue
                    if training_receipt(root,plan,job) is None:
                        from chapter3_bser.learning.train import Trainer
                        trainer=Trainer(config,root/job['train'],mechanism=plan['mechanism_only'])
                        summary=trainer.run();_record_training(root,plan,job,trainer,summary)
                else:
                    from chapter3_bser.learning.evaluation_records import verified_rows
                    output=root/job['evaluation'];trained=None if job['seed'] is None else training_receipt(root,plan,job)
                    if job['seed'] is not None and trained is None:raise ValueError('verified native training required: '+job['name'])
                    if output.exists():
                        _,receipt=verified_rows(output/'episodes.jsonl',plan['source_review'])
                        if (receipt.get('map_condition')!=job['map_condition'] or receipt['completed']!=plan['spec']['evaluation_count']
                                or receipt['manifest_sha256']!=digest(read(root/'inputs/evaluation.json'))
                                or trained is not None and receipt['policy_id']!=trained['model_state_sha256']):raise ValueError('existing map evaluation differs from plan')
                    elif job['seed'] is None:
                        from chapter3_bser.learning.reference import evaluate_reference
                        evaluate_reference(config,root/'inputs/evaluation.json',output,mechanism=plan['mechanism_only'])
                    else:
                        from chapter3_bser.learning.evaluate import evaluate
                        evaluate(root/job['train']/'checkpoint_final.pt',root/'inputs/evaluation.json',output,mechanism=plan['mechanism_only'])
                results.append(dict(job=job['name'],stage=phase,status='completed'))
    require_review(plan['source_review'])
    return dict(executed=True,jobs=results,performance_passed=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare');prep.add_argument('--root',required=True);prep.add_argument('--spec',default=str(ROOT/'configs/map_comparison_v1.json'))
    prep.add_argument('--mechanism-check',action='store_true')
    prep.add_argument('--exclude-run',dest='exclude_runs',action='append')
    launch=subs.add_parser('run');launch.add_argument('--root',required=True);launch.add_argument('--stage',choices=('train','diagnose','evaluate','all'),default='all')
    launch.add_argument('--groups',nargs='+',choices=GROUPS);launch.add_argument('--seeds',nargs='+',type=int);launch.add_argument('--validation')
    summary=subs.add_parser('summarize');summary.add_argument('--root',required=True);summary.add_argument('--output',required=True)
    for sub in (prep,launch,summary):sub.add_argument('--execute',action='store_true')
    args=vars(p.parse_args(argv));execute=args.pop('execute');command=args.pop('command')
    if not execute:result=dict(executed=False,command=command,**args)
    elif command=='prepare':result=prepare(args['root'],read(args['spec']),mechanism=args['mechanism_check'],exclude_runs=args['exclude_runs'])
    elif command=='run':result=run(**args)
    else:
        from chapter3_bser.learning.map_report import summarize
        result=summarize(**args)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()
