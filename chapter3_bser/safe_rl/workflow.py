"""Manual OLD safety-only learning/evaluation, explicit --execute and validation."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from chapter3_bser.learning.config import validate as validate_task,scenario_fingerprint
from chapter3_bser.learning.checkpoint import state_digest
from chapter3_bser.learning.provenance import verify_reviewed_changes,research_readiness,require_review
from chapter3_bser.learning.map_suite import require_linux_validation
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.safe_rl.options import validate,SCHEMA
from chapter3_bser.safe_rl.learner import SafetyLearner
from chapter3_bser.safe_rl.runtime import SafetyRuntime
from chapter3_bser.safe_rl import dataset


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def admission(validation):
    identity=verify_reviewed_changes();research_readiness(identity)
    path=Path(validation)
    if path.is_dir():path=path/'validation.json'
    require_linux_validation(path,identity)
    return identity


def inputs(config_path,manifest_path,*,training,episodes,cases=None):
    c=validate_task(read(config_path));manifest=read(manifest_path);scenes=manifest.get('scenarios')
    if not isinstance(scenes,list) or not scenes:raise ValueError('nonempty frozen scenario manifest required')
    if c.get('map_experiment',{}).get('map_id','OLD')!='OLD' or c.get('map_experiment',{}).get('research_config') is not None:
        raise ValueError('safety study must use OLD')
    if c.get('map_experiment',{}).get('executor_safety',{}).get('enabled'):raise ValueError('disable executor-only guards')
    if training:
        if any(s.get('scenario_role')!='train' or s.get('scenario_split')!='train' for s in scenes):
            raise ValueError('training requires explicit train/train scenarios; no development/evaluation reuse')
    elif any(s.get('scenario_role') not in ('eval','validation') or s.get('scenario_split')!=s.get('scenario_role') for s in scenes):
        raise ValueError('evaluation must be held out from training')
    normalized=copy.deepcopy(manifest)
    if not training:
        for scene in normalized['scenarios']:scene.update(scenario_role='eval',scenario_split='eval')
    from chapter3_bser.learning.scenarios import validate_manifest
    validate_manifest(normalized,c,evaluation=not training)
    if cases:
        scenes=[s for s in scenes if any(str(s['scenario_id']).endswith('_'+x) for x in cases)]
        if len(scenes)!=len(cases):raise ValueError('each requested case must match exactly once')
    if type(episodes) is not int or episodes<1:raise ValueError('positive explicit episode count required')
    if not training and episodes!=len(scenes):raise ValueError('evaluation episode count must equal selected frozen scenes')
    return c,scenes


def execute(args):
    if args.command=='train' or args.mode=='learned' or args.policy:
        raise ValueError('safety learning admission suspended: audit findings3/4/6 remain unresolved; use rule evaluation')
    identity=admission(args.validation);options=validate(read(args.options) if args.options else None)
    torch.set_num_threads(1)
    training=args.command=='train';collecting=args.command=='collect'
    c,scenes=inputs(args.config,args.scenarios,training=training or collecting,episodes=args.episodes,cases=args.cases)
    output=Path(args.output).resolve()
    if not output.is_relative_to(ROOT/'work'/'collision_terminal'):raise ValueError('outputs must be fresh under work/collision_terminal')
    if output.exists():raise FileExistsError(output)
    learner=None if args.mode in ('rule','nominal') else SafetyLearner(options)
    training_ids=[];training_fingerprints=[]
    if args.policy:
        if training:raise ValueError('training checkpoint resume is unsupported')
        payload=torch.load(args.policy,map_location='cpu',weights_only=False)
        if payload.get('schema')!=SCHEMA or payload.get('source_identity')!=identity or payload.get('resume_supported') is not False:
            raise ValueError('only this new safety policy identity may be evaluated')
        if payload.get('safety_state_sha256')!=state_digest(payload['safety_state']):raise ValueError('safety policy bytes changed')
        if learner is None:raise ValueError('rule control cannot load learned weights')
        training_ids=payload['training_scenario_ids'];training_fingerprints=payload['training_fingerprints']
        if set(training_ids)&{s['scenario_id'] for s in scenes} or set(training_fingerprints)&{scenario_fingerprint(s) for s in scenes}:
            raise ValueError('evaluation scenarios overlap safety training')
        learner.load_for_evaluation(payload['safety_state'])
    elif not training and args.mode=='learned':raise ValueError('learned evaluation requires new safety --policy')
    if training and args.mode!='learned':raise ValueError('train is safety-only learned mode; task B2/B3 training remains disabled')
    offline_episodes=[]
    if training:
        if not args.risk_data:raise ValueError('train requires independent train-split factual --risk-data; never reuse0002/0006 development diagnoses')
        offline_episodes,offline_scenes=dataset.load_collection(args.risk_data,identity,options)
        training_ids.extend(s['scenario_id'] for s in offline_scenes)
        training_fingerprints.extend(scenario_fingerprint(s) for s in offline_scenes)
        for transitions in offline_episodes:learner.add_episode(transitions)
        if len(learner.replay)<options['warmup_samples']:raise ValueError('insufficient factual risk pretraining transitions')
        if args.pretrain_updates<options['minimum_updates']:raise ValueError('explicit pretrain update count below readiness threshold')
    output.mkdir(parents=True,exist_ok=False)
    run=dict(schema=SCHEMA,source_identity=identity,options=options,config=c,mode=args.mode,
        command=args.command,scenario_manifest_sha256=sha(args.scenarios),scenario_ids=[s['scenario_id'] for s in scenes],
        task_policy='D2_ZERO_no_task_network',task_training=False,checkpoint_resume=False,
        learned_safety_only=training,policy_sha256=sha(args.policy) if args.policy else None,
        unprotected_original_nominal_collection=collecting,risk_data_receipt_sha256=sha(Path(args.risk_data)/'dataset.json') if training else None,
        performance_passed=False,safety_guarantee=False)
    write(output/'run.json',run);write(output/'scenarios.json',dict(scenarios=scenes))
    episodes=[];global_step=0;metrics=[];dataset_files=[]
    if training:
        for _ in range(args.pretrain_updates):metrics.append(learner.update())
    with (output/'episodes.jsonl').open('x',encoding='utf-8') as stream:
        for episode in range(args.episodes):
            require_review(identity);scene=copy.deepcopy(scenes[episode%len(scenes)])
            innovation_seed=int(c['seed'])+episode if training else int(scene['scenario_seed'])
            runtime=SafetyRuntime(c,scene,seed=innovation_seed,episode_id=episode+1,options=options,learner=learner,collection_only=collecting)
            try:
                while not runtime.terminal:
                    runtime.advance(ZeroResidualPolicy(),explore=False);global_step+=1
                row=runtime.summary()
                if collecting:dataset_files.append(dataset.save(output/f'factual_episode_{episode+1:06d}.npz',runtime.safety_transitions))
                if training:
                    training_ids.append(scene['scenario_id']);training_fingerprints.append(scenario_fingerprint(scene))
                    learner.add_episode(runtime.safety_transitions)
                    # Exactly one independent safety update per scheduled block;
                    # does not invoke/alter task model update schedules.
                    updates=max(1,len(runtime.safety_transitions)//options['update_frequency'])
                    for _ in range(updates):
                        metric=learner.update()
                        if metric is not None:metrics.append(metric)
            finally:runtime.close()
            require_review(identity);row.update(episode=episode+1,safety_training=training)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();episodes.append(row)
    if training:
        state=learner.state();payload=dict(schema=SCHEMA,source_identity=identity,safety_state=state,safety_state_sha256=state_digest(state),
            training_scenario_ids=sorted(set(training_ids)),training_fingerprints=sorted(set(training_fingerprints)),
            task_policy='D2_ZERO_no_task_network',resume_supported=False,formal_use_allowed=False,performance_passed=False)
        if learner.ready:torch.save(payload,output/'safety_policy.pt')
        write(output/'learning_metrics.json',dict(updates=learner.updates,policy_ready=learner.ready,metrics=metrics))
    if collecting:write(output/'dataset.json',dict(schema=SCHEMA,source_identity=identity,complete=True,files=dataset_files,
        run_sha256=sha(output/'run.json'),scenarios_sha256=sha(output/'scenarios.json')))
    result=dict(schema=SCHEMA,source_identity=identity,completed=True,episodes=len(episodes),physical_steps=global_step,
        collisions=sum(r['termination_reason']=='obstacle_collision' for r in episodes),successes=sum(bool(r['success']) for r in episodes),
        safety_updates=learner.updates if learner is not None else 0,task_updates=0,performance_passed=False,safety_guarantee=False)
    require_review(identity);write(output/'completed.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    for name in ('train','evaluate','collect'):
        sub=subs.add_parser(name);sub.add_argument('--config',required=True);sub.add_argument('--scenarios',required=True)
        sub.add_argument('--output',required=True);sub.add_argument('--validation',required=True)
        sub.add_argument('--episodes',type=int,required=True);sub.add_argument('--mode',choices=('nominal',) if name=='collect' else ('rule','learned'),default='nominal' if name=='collect' else 'learned' if name=='train' else 'rule')
        sub.add_argument('--options');sub.add_argument('--policy');sub.add_argument('--cases',nargs='+')
        sub.add_argument('--risk-data');sub.add_argument('--pretrain-updates',type=int,default=256)
        sub.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    result=execute(args) if args.execute else dict(executed=False,algorithm=SCHEMA,**vars(args))
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()
