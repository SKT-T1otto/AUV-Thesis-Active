"""Explicit bounded core-map mechanism run, never training or performance acceptance."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
from project_paths import ROOT


def run(output, *, scene_indices=(0,3), max_steps=32):
    import torch
    from chapter3_bser.learning.runtime import RepairedMissionRuntime
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review
    from chapter3_bser.experiments.coupled_search_v1.core_config import CORE_DEFAULT
    if not 1<=max_steps<=32 or not 1<=len(scene_indices)<=3:raise ValueError('bounded mechanism only')
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):raise ValueError('output must be a new Active/work directory')
    identity=verify_reviewed_changes();torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    scenes=json.loads((ROOT/'verification/mechanism_scenes.json').read_text())['scenarios']
    results=[];started=0
    try:
        for index in scene_indices:
            scene=copy.deepcopy(scenes[index]);scene['max_steps']=max_steps
            config=json.loads((ROOT/'configs/b2_mechanism.json').read_text());config['max_steps']=max_steps
            extension=copy.deepcopy(CORE_DEFAULT);extension['enabled']=True
            started+=1
            runtime=RepairedMissionRuntime(config,scene,seed=82729,episode_id=index,map_research=extension)
            try:
                cursor=0
                with (output/f'public_events_{index}.jsonl').open('x',encoding='utf-8') as stream:
                    while True:
                        for event in runtime.env.read_public_events(cursor):
                            stream.write(json.dumps(asdict(event),allow_nan=False)+'\n');cursor=event.sequence
                        if runtime.terminal:break
                        if runtime.step>=max_steps:raise RuntimeError('mechanism physical-step overrun')
                        runtime.advance(ZeroResidualPolicy(),explore=False)
                session=runtime.map_research_session;summary=session.summary()
                records=list(session.records)
                (output/f'audit_{index}.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
                (output/f'allocator_{index}.json').write_text(json.dumps(runtime.controller.allocator.audit,indent=2),encoding='utf-8')
                if not summary['valid']:raise RuntimeError('public event adapter invalid')
                if index==0 and summary['counts'].get('core_proposals',0)<1:
                    raise RuntimeError('core branch only fell back; no coupled proposal was used')
                if summary['counts'].get('physical_transitions')!=runtime.step:raise RuntimeError('incomplete transition evidence')
                results.append(dict(scene_index=index,steps=runtime.step,summary=summary,runtime=runtime.summary()))
            finally:runtime.close()
        require_review(identity)
        result=dict(passed=True,source_identity=identity,episodes_started=started,episodes=results,
            physical_steps=sum(r['steps'] for r in results),model_restored=False,optimizer_updates=0,
            training_started=False,performance_passed=False)
    except Exception:
        import traceback
        result=dict(passed=False,source_identity=identity,episodes_started=started,completed_episodes=results,
                    traceback=traceback.format_exc(),training_started=False,performance_passed=False)
    (output/'mechanisms.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--scenes',type=int,nargs='+',default=[0,3]);p.add_argument('--max-steps',type=int,default=32)
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    if args.execute:
        result=run(args.output,scene_indices=tuple(args.scenes),max_steps=args.max_steps)
        print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','completed_episodes')},indent=2))
        if not result['passed']:raise SystemExit(1)
    else:print(json.dumps(dict(executed=False,scene_indices=args.scenes,max_steps=args.max_steps,output=args.output)))


if __name__=='__main__':main()
