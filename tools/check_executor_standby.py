"""Eight ZERO episodes x <=32 steps; no training or checkpoint-file restore."""
import argparse
import copy
import json
from pathlib import Path
import traceback
from project_paths import ROOT


def run(output, *, selection=None):
    import torch
    from core.scenarios.ch3_generator_impl import build_scenario_manifests
    from chapter3_bser.learning.config import validate
    from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review, training_admission
    from chapter3_bser.learning.checkpoint import write_json
    from chapter3_bser.learning.runtime import RepairedMissionRuntime
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.learning.evaluation_records import metric_row
    from chapter3_bser.learning import map_contract
    from chapter3_bser.learning.standby_experiment import episode_record
    from chapter3_bser.safety.executor_standby import OPTIONS
    source=verify_reviewed_changes();output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):raise ValueError('fresh Active/work output required')
    output.mkdir(parents=True,exist_ok=False)
    scenes=build_scenario_manifests(count=12,generator_seed=827699,split='validation',profiles=['M20_MOVING_UNKNOWN_MULTI'])['M20_MOVING_UNKNOWN_MULTI']['scenarios']
    settings=json.loads((ROOT/'configs/executor_standby_pilot_v1.json').read_text(encoding='utf-8'))
    base=json.loads((ROOT/'configs/b2_mechanism.json').read_text(encoding='utf-8'))
    all_cases=[(m,e,i) for m in ('OLD','NEW') for e in (False,True) for i in (5,10)]
    cases=all_cases if selection is None else list(selection)
    if not cases or len(set(cases))!=len(cases) or any(c not in all_cases for c in cases):raise ValueError('unique bounded standby cases required')
    rows=[];started=0;physical=0;previous=torch.get_num_threads();torch.set_num_threads(1)
    result=dict(passed=False,source=source,episodes_started=0,max_episode_steps=32,
        max_complete_updates_per_model=0,temporary_models=0,checkpoint_file_restore_executed=False,
        requested_cases=cases,
        scope='Selected reused development scenes (0006,0011), 32-step truncations, original/bounded map x guard off/on; no long-horizon success claim.')
    try:
        for map_id,enabled,index in cases:
            scene=copy.deepcopy(scenes[index]);scene.update(max_steps=32,scenario_role='eval',scenario_split='eval')
            scene['source_scenario_id']=scene['scenario_id']
            scene['scenario_id']='eval_gen827699__'+scene['scenario_id']
            config=copy.deepcopy(base)
            config.update(max_steps=32,total_main_trajectories=1,max_total_environment_steps=32,
                output_dir=str(output/f'{map_id}_{enabled}_{index}/train'),map_experiment=dict(
                    schema=map_contract.SCHEMA,map_id=map_id,research_config=None if map_id=='OLD' else settings['new_map_config'],
                    protocol_sha256='e'*64,executor_safety=dict(OPTIONS,enabled=enabled)))
            config=validate(config);training_admission(mechanism=True,config=config)
            started+=1
            print(f'[standby probe] {map_id} guard={enabled} scene={index+1:04}',flush=True)
            runtime=RepairedMissionRuntime(config,scene,seed=int(scene['scenario_seed']),map_research=map_contract.research_config(config))
            try:
                while not runtime.terminal:
                    map_contract.advance(runtime,ZeroResidualPolicy(),explore=False);physical+=1
                    ctx=runtime.guidance;e=ctx.agent_assignments[3];view=ctx.executor_assignment
                    if (e.tracking_waypoint,e.hold_state,e.reachable,e.planned_path)!=(view.tracking_waypoint,view.hold_state,view.reachable,view.planned_path):
                        raise ValueError('inconsistent executor guidance views')
                row=metric_row(runtime,config,scene,policy_id='standby_zero_mechanism',method='D2',identity=source,mechanism=True)
                rows.append(row)
                write_json(output/f'episode_{len(rows):02}.json',row)
            finally:runtime.close()
        if started!=len(cases) or physical>256:raise ValueError('mechanism budget changed')
        if any(r['actual_length']>32 for r in rows):raise ValueError('episode horizon exceeded')
        require_review(source)
        result.update(passed=True,episodes_started=started,physical_steps=physical,
            additional_physical_steps=physical,summary=[episode_record('guard_on' if r['map_condition']['executor_safety']['enabled'] else 'guard_off',
                r['map_condition']['map_id']+'_ZERO',r) for r in rows],
            guard_executor_search_collisions=sum(3 in r['first_collision_agent_ids'] and r['first_collision_phase']=='Search'
                for r in rows if r['map_condition']['executor_safety']['enabled']))
    except Exception:
        result.update(episodes_started=started,physical_steps=physical,traceback=traceback.format_exc())
    finally:
        torch.set_num_threads(previous);write_json(output/'standby_workflow.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv);result=run(args.output) if args.execute else dict(executed=False,output=args.output)
    print(json.dumps(result,indent=2))
    if args.execute and not result['passed']:raise SystemExit(1)
    return result


if __name__=='__main__':main()
