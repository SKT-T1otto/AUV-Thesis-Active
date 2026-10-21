"""Eight bounded native-map episodes; fresh models, no checkpoint-file restore."""
import argparse
import copy
import json
from pathlib import Path
import traceback
from project_paths import ROOT


def continuous_health(row):
    health=row['map_metrics'];found=row['found_step']
    # Discovery legitimately ends search. Require late participation when the
    # search lasts that long; never force continued searching after discovery.
    expected=min(row['actual_length'],found if found is not None else row['actual_length'])
    return (health['search_batches']==expected and health['unavailable_search_batches']==0
            and health['available_search_batches']==expected
            and health['core_scored_steps']>=max(0,expected-1)
            and (expected<24 or health['last_core_scored_step']>=expected-1))


def run(output,*,groups=('OLD_B2','NEW_B2','OLD_B3','NEW_B3')):
    import torch
    from chapter3_bser.learning.map_suite import prepare,load_plan,_record_training,read
    from chapter3_bser.learning.map_report import summarize
    from chapter3_bser.learning.train import Trainer
    from chapter3_bser.learning.evaluate import _evaluate_loaded,preserved_rng
    from chapter3_bser.learning.map_contract import metadata
    from chapter3_bser.learning.checkpoint import state_digest,write_json
    from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):raise ValueError('new Active/work output required')
    output.mkdir(parents=True,exist_ok=False);source=verify_reviewed_changes()
    spec=read(ROOT/'configs/map_comparison_v1.json')
    spec.pop('convergence',None)
    spec.update(groups=list(groups),seeds=[2729],train_episodes=1,max_steps=32,evaluation_count=1,
                checkpoint_interval=1,bootstrap_draws=32)
    scenes=read(ROOT/'verification/mechanism_scenes.json')['scenarios']
    normal=copy.deepcopy(scenes[0]);normal['max_steps']=32
    challenge=read(ROOT/'configs/map_persistence_cases.json')['scenarios'][0]
    challenge=copy.deepcopy(challenge);challenge.update(max_steps=32,scenario_role='eval',scenario_split='eval',
        scenario_id='map_persistence_eval',scenario_seed=910828)
    training=dict(scenario_profile=spec['new_map_config'].get('profile','M20_MOVING_UNKNOWN_MULTI'),scenario_split='train',scenario_role='train',scenario_count=1,scenarios=[normal])
    evaluation=dict(scenario_count=1,scenarios=[challenge]);root=output/'collision_terminal/suite'
    started=0;completed=[]
    try:
        prepare(root,spec,mechanism=True,training_manifests={2729:training},evaluation_manifest=evaluation)
        plan=load_plan(root)
        previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for job in plan['jobs']:
                config=read(root/job['config']);started+=1
                trainer=Trainer(config,root/job['train'],mechanism=True);summary=trainer.run()
                receipt=_record_training(root,plan,job,trainer,summary)
                if summary['optimizer_updates']!=4:raise ValueError('expected four complete multi-agent updates')
                model=trainer.model
                payload=dict(model_state_sha256=state_digest(model.training_state_dict()),training_scenario_ids=sorted(trainer.training_scenario_ids),
                    training_scenario_seeds=sorted(trainer.training_scenario_seeds),training_scenario_fingerprints=sorted(trainer.training_scenario_fingerprints),
                    **metadata(config))
                started+=1
                with preserved_rng():
                    _evaluate_loaded(model,config,payload,root/'inputs/evaluation.json',root/job['evaluation'],identity=source,mechanism=True)
                after=state_digest(model.training_state_dict())
                if after!=payload['model_state_sha256']:raise ValueError('evaluation changed frozen policy')
                train_row=json.loads((root/job['train']/'episodes.jsonl').read_text(encoding='utf-8').splitlines()[0])
                eval_row=json.loads((root/job['evaluation']/'episodes.jsonl').read_text(encoding='utf-8').splitlines()[0])
                if job['map_id']=='NEW' and train_row['map_metrics']['core_proposals']<1:raise ValueError('NEW only fell back in normal training fixture')
                if job['map_id']=='NEW':
                    for phase,row in (('train',train_row),('evaluation',eval_row)):
                        if not continuous_health(row):
                            raise ValueError('continuous new-map participation failed in '+phase)
                    if train_row['map_metrics']['last_core_scored_step']<24:
                        raise ValueError('long-search fixture must exercise late new-map scoring')
                completed.append(dict(job=job['name'],map_condition=job['map_condition'],training_steps=summary['environment_steps'],
                    evaluation_steps=eval_row['actual_length'],complete_updates=summary['optimizer_updates'],initial_state=trainer.initial_model_state_sha256,
                    core_proposals=train_row['map_metrics']['core_proposals'],evaluation_policy_unchanged=True,
                    training_map_health=train_row['map_metrics'],evaluation_map_health=eval_row['map_metrics']))
        finally:torch.set_num_threads(previous_threads)
        report=summarize(root,output/'report')
        if not report['complete'] or len(report['groups'])!=len(groups):raise ValueError('native map report incomplete')
        require_review(source)
        result=dict(passed=True,source=source,episodes_started=started,training_episodes=len(groups),evaluation_episodes=len(groups),
            physical_steps=sum(r['training_steps']+r['evaluation_steps'] for r in completed),max_episode_steps=32,
            max_complete_updates_per_model=4,checkpoint_file_restore_executed=False,formal_training_started=False,
            jobs=completed,report=report,performance_passed=False)
    except Exception:
        result=dict(passed=False,source=source,episodes_started=started,jobs=completed,traceback=traceback.format_exc(),
                    checkpoint_file_restore_executed=False,formal_training_started=False,performance_passed=False)
    write_json(output/'map_workflow.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not args.execute:print(json.dumps(dict(executed=False,episodes=8,max_steps=32,max_complete_updates_per_model=4,output=args.output)));return
    result=run(args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','report')},indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
