"""Eight OLD-map ZERO episodes, <=32 steps each; no checkpoint restore."""
import argparse
import copy
import json
from pathlib import Path
import traceback
from project_paths import ROOT


def run(output):
    import torch
    from core.scenarios.ch3_generator_impl import build_scenario_manifests
    from chapter3_bser.learning.config import validate
    from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review,training_admission
    from chapter3_bser.learning.checkpoint import write_json
    from chapter3_bser.learning.runtime import RepairedMissionRuntime
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.learning.evaluation_records import metric_row
    from chapter3_bser.learning.executor_experiment import record
    from chapter3_bser.learning import map_contract
    from chapter3_bser.safety.executor_recovery_options import OPTIONS
    source=verify_reviewed_changes();output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):raise ValueError('fresh Active/work output required')
    output.mkdir(parents=True,exist_ok=False)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    result=dict(passed=False,source=source,episodes_started=0,physical_steps=0,
        max_episode_steps=32,max_complete_updates_per_model=0,temporary_models=0,
        checkpoint_file_restore_executed=False,old_map_only=True,
        scope='Reused developer scenes 0002/0006/0011/0012, truncated to 32 steps; no long-horizon safety or learned-policy acceptance.')
    started=0;physical=0;rows=[]
    try:
        scenes=build_scenario_manifests(count=12,generator_seed=827699,split='validation',profiles=['M20_MOVING_UNKNOWN_MULTI'])['M20_MOVING_UNKNOWN_MULTI']['scenarios']
        base=json.loads((ROOT/'configs/b2_mechanism.json').read_text(encoding='utf-8'))
        for enabled in (False,True):
            for index in (1,5,10,11):
                scene=copy.deepcopy(scenes[index]);scene.update(max_steps=32,scenario_role='eval',scenario_split='eval')
                scene['source_scenario_id']=scene['scenario_id'];scene['scenario_id']='eval_gen827699__'+scene['scenario_id']
                config=copy.deepcopy(base)
                config.update(max_steps=32,total_main_trajectories=1,max_total_environment_steps=32,
                    output_dir=str(output/f'{enabled}_{index}/train'),map_experiment=dict(schema=map_contract.SCHEMA,
                        map_id='OLD',research_config=None,protocol_sha256='d'*64,executor_safety=dict(OPTIONS,enabled=enabled)))
                config=validate(config);training_admission(mechanism=True,config=config);started+=1
                print(f'[old executor] recovery={enabled} scene={index+1:04}',flush=True)
                runtime=RepairedMissionRuntime(config,scene,seed=int(scene['scenario_seed']),map_research=None)
                try:
                    if runtime.map_research_session is not None:raise ValueError('new map unexpectedly active')
                    while not runtime.terminal:
                        t=runtime.step
                        try:transition=map_contract.advance(runtime,ZeroResidualPolicy(),explore=False)
                        finally:physical+=runtime.step-t
                        if enabled and not runtime.env.get_task_state().target_found and transition['action_enabled'][3]:
                            raise ValueError('search executor residual gate was not disabled')
                        a=runtime.guidance.agent_assignments[3];b=runtime.guidance.executor_assignment
                        if (a.tracking_waypoint,a.hold_position,a.hold_state,a.reachable)!=(b.tracking_waypoint,b.hold_position,b.hold_state,b.reachable):
                            raise ValueError('executor guidance views inconsistent')
                    row=metric_row(runtime,config,scene,policy_id='old_executor_zero_mechanism',method='D2',identity=source,mechanism=True)
                    rows.append(row);write_json(output/f'episode_{len(rows):02}.json',row)
                finally:runtime.close()
        if started!=8 or physical>256 or any(r['actual_length']>32 for r in rows):raise ValueError('mechanism budget exceeded')
        require_review(source)
        baseline={r['scenario_id']:r for r in rows[:4]}
        regressions=[r['scenario_id'] for r in rows[4:] if r['termination_reason']=='obstacle_collision'
            and baseline[r['scenario_id']]['termination_reason']!='obstacle_collision']
        result.update(passed=True,team_collision_regressions=regressions,
            performance_passed=False,summary=[record('recovery' if r['map_condition']['executor_safety']['enabled'] else 'baseline',r) for r in rows])
    except Exception:result['traceback']=traceback.format_exc()
    finally:
        torch.set_num_threads(previous);result.update(episodes_started=started,physical_steps=physical,additional_physical_steps=physical)
        write_json(output/'executor_workflow.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv);result=run(args.output) if args.execute else dict(executed=False,output=args.output)
    print(json.dumps(result,indent=2))
    if args.execute and not result['passed']:raise SystemExit(1)
    return result


if __name__=='__main__':main()
