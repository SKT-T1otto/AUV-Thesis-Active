"""Six bounded paired OLD ZERO prefixes, <=32 steps, no models/restores."""
import argparse
import copy
import json
from pathlib import Path
import traceback
from project_paths import ROOT
from tools.executor_trace import run_episode, read, write, behavioral_summary, sha


def run(output):
    from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review, training_admission
    from chapter3_bser.learning.config import validate
    from chapter3_bser.learning.map_contract import SCHEMA
    from chapter3_bser.safety.executor_standby import OPTIONS
    from core.scenarios.ch3_generator_impl import build_scenario_manifests
    source=verify_reviewed_changes();output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):
        raise ValueError('bounded diagnostic output must stay in Active/work')
    output.mkdir(parents=True,exist_ok=False)
    scenes=build_scenario_manifests(count=12,generator_seed=827699,split='validation',profiles=['M20_MOVING_UNKNOWN_MULTI'])['M20_MOVING_UNKNOWN_MULTI']['scenarios']
    starts,physical,pairs=0,0,[]
    result=dict(passed=False,source=source,planned_environment_initializations=6,
        max_episode_steps=32,max_complete_updates_per_model=0,temporary_models=0,
        checkpoint_file_restore_executed=False,training_executed=False,
        full_82_96_step_reproduction_executed=False,
        scope='Paired plain/recorded OLD prefixes: original 0002/0006 and v1 0011; no v2 control or long-horizon performance evidence')
    try:
        for index,guard in ((1,False),(5,False),(10,True)):
            scene=copy.deepcopy(scenes[index]);scene.update(max_steps=32,scenario_role='eval',scenario_split='eval')
            scene['source_scenario_id']=scene['scenario_id'];scene['scenario_id']='eval_gen827699__'+scene['scenario_id']
            c=read(ROOT/'configs/b2_mechanism.json')
            c.update(max_steps=32,total_main_trajectories=1,max_total_environment_steps=32,
                output_dir=str(output/'unused_train'),map_experiment=dict(schema=SCHEMA,map_id='OLD',research_config=None,
                    protocol_sha256='e'*64,executor_safety=dict(OPTIONS,enabled=guard)))
            c=validate(c);training_admission(mechanism=True,config=c)
            values=[]
            name=('guard_v1_' if guard else 'baseline_')+f'{index+1:04}'
            for instrumented in (False,True):
                starts+=1
                write(output/f'initialization_{starts:02}.json',dict(index=starts,case=name,instrumented=instrumented,max_steps=32))
                print(f'[diagnostic prefix {starts}/6] {name} recorded={instrumented}',flush=True)
                directory=output/name/('recorded' if instrumented else 'plain')
                summary,signatures=run_episode(c,scene,directory,instrumented=instrumented)
                physical+=summary['actual_length'];values.append((summary,signatures))
                if instrumented:
                    from tools.executor_trace_analysis import analyze_episode
                    analysis=analyze_episode(directory,scene,output/name/'posthoc')
                    if (analysis['motion_audit']['max_prior_reconstruction_error']>2e-6
                        or analysis['motion_audit']['actual_navigation_mismatch_steps']
                        or analysis['motion_audit']['nonzero_learned_residual_steps']):
                        raise ValueError('actual physical-input audit failed')
            summary_equal=behavioral_summary(values[0][0])==behavioral_summary(values[1][0])
            signatures_equal=values[0][1]==values[1][1]
            pairs.append(dict(name=name,exact_step_signature_parity=signatures_equal,behavioral_summary_parity=summary_equal,
                ignored_wall_timer_fields=['compile_seconds','max_compile_seconds'],
                passed=summary_equal and signatures_equal,physical_steps=sum(v[0]['actual_length'] for v in values)))
        require_review(source)
        result.update(passed=all(p['passed'] for p in pairs),pairs=pairs)
    except Exception:
        result['traceback']=traceback.format_exc()
    result.update(environment_initializations=starts,physical_steps=physical,additional_physical_steps=physical)
    if starts>8 or physical>256:
        result.update(passed=False,budget_violation=True)
    write(output/'diagnostic_workflow.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv);result=run(args.output) if args.execute else dict(executed=False,output=args.output)
    print(json.dumps(result,indent=2))
    if args.execute and not result['passed']:raise SystemExit(1)
    return result


if __name__=='__main__':main()
