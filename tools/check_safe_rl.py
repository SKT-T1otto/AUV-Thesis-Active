"""Seven bounded episodes: same-host exact repeat, no cross-host hash admission."""
import argparse,copy,json,traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review,training_admission
from chapter3_bser.learning.config import validate as validate_task
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.safe_rl.runtime import SafetyRuntime
from chapter3_bser.safe_rl.options import validate
from chapter3_bser.safe_rl.learner import SafetyLearner
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from tools.executor_trace import signature
from tools.audit_safe_speed import run as speed_audit
from tools.safe_rl_validation import require_same_host_repeat,archived_report,environment

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def fault_guidance(context,goal,positions):
    assignments=tuple(replace(x,final_waypoint=tuple(goal),planned_path=(tuple(positions[x.agent_id]),tuple(goal)),
        tracking_waypoint=tuple(goal),hold_state=False,reachable=True) for x in context.agent_assignments)
    return replace(context,agent_assignments=assignments)

class FaultBridge:
    def __init__(self,inner,goal,env):self.inner=inner;self.goal=goal;self.env=env
    def __getattr__(self,name):return getattr(self.inner,name)
    def compile_guidance(self,*args,**kwargs):
        return fault_guidance(self.inner.compile_guidance(*args,**kwargs),self.goal,self.env.get_agent_state().positions)

def fixture_learning(options):
    # Four in-memory synthetic updates; clearly separate from research learning.
    options=dict(options,batch_size=4,warmup_samples=4,minimum_updates=4)
    learner=SafetyLearner(options);transitions=[]
    for t in range(8):
        transitions.append(dict(safety_features=np.full((4,192),t/20,np.float32),
            normalized_executed_acceleration=np.zeros((4,3),np.float32),
            collision_cost=np.array([False,t==7,False,False],np.float32),
            safety_rows=[dict(mode='NOMINAL' if t==5 else 'BRAKE',model_checked=True) for _ in range(4)]))
    learner.add_episode(transitions)
    metrics=[learner.update() for _ in range(4)]
    return learner,options,metrics

def run(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):raise ValueError('bounded output must stay under work')
    output.mkdir(parents=True,exist_ok=False);source=verify_reviewed_changes();options=validate()
    from tools.safe_rl_validation_source import verify as verify_runtime_bytes
    protection=verify_runtime_bytes(ROOT)
    torch.set_num_threads(1)
    generated=build_scenario_manifests(count=12,generator_seed=827699,split='validation',profiles=['M20_MOVING_UNKNOWN_MULTI'])['M20_MOVING_UNKNOWN_MULTI']['scenarios']
    planned=[('baseline_0002',1,'baseline'),('baseline_repeat_0002',1,'repeat'),('composition_0002',1,'collection'),('rule_0002',1,'rule'),
             ('rule_0006',5,'rule'),('fault_all_four',1,'fault'),('learned_fixture_0006',5,'learned')]
    result=dict(passed=False,source=source,planned_environment_initializations=len(planned),max_episode_steps=32,
        checkpoint_file_restore_executed=False,max_complete_updates_per_model=4,task_training=False,
        research_training=False,full_82_96_step_regression=False,safety_guarantee=False,performance_passed=False,
        environment=environment(),parent_runtime_byte_protection=protection)
    starts=steps=0;records=[];baseline=None;baseline_signatures=[]
    try:
        for name,index,mode in planned:
            require_review(source);scene=copy.deepcopy(generated[index]);scene.update(max_steps=32,scenario_role='eval',scenario_split='eval')
            scene['scenario_id']='eval_gen827699__'+scene['scenario_id']
            config=json.loads((ROOT/'configs/b2_mechanism.json').read_text())
            config.update(max_steps=32,total_main_trajectories=1,max_total_environment_steps=32,
                output_dir=str(output/'unused_task_training'))
            config=validate_task(config);training_admission(mechanism=True,config=config)
            directory=output/name;directory.mkdir();learner=None;local_options=options
            if mode=='learned':
                learner,local_options,metrics=fixture_learning(options);write(directory/'synthetic_updates.json',dict(metrics=metrics,not_research_training=True))
            starts+=1;write(output/f'initialization_{starts:02d}.json',dict(case=name,max_steps=32,source=source))
            print(f'[safe-RL mechanism {starts}/{len(planned)}] {name}',flush=True)
            runtime=None;frames=[];signatures=[]
            try:
                if mode in ('baseline','repeat'):runtime=RepairedMissionRuntime(config,scene,seed=int(scene['scenario_seed']))
                else:runtime=SafetyRuntime(config,scene,seed=int(scene['scenario_seed']),options=local_options,learner=learner,collection_only=mode=='collection')
                if mode in ('baseline','repeat'):signatures.append(signature(runtime))
                if mode=='fault':
                    # True geometry ONLY chooses an offline fault input. Controller
                    # receives the point, never the obstacle or its dimensions.
                    goal=tuple(scene['obstacles'][0]['center'])
                    runtime.safety_bridge.inner=FaultBridge(runtime.safety_bridge.inner,goal,runtime.env)
                    runtime.guidance=fault_guidance(runtime.guidance,goal,runtime.env.get_agent_state().positions)
                    runtime.env.install_guidance(runtime.guidance);runtime.observations=runtime.env.refresh_observation_after_guidance()
                while not runtime.terminal:
                    row=runtime.advance(ZeroResidualPolicy(),explore=False);steps+=1
                    public=runtime.env.get_agent_state()
                    frames.append(dict(t=row['t'],positions=public.positions,velocities=public.velocities,rewards=row['rewards'].tolist(),
                        tracking=[x.tracking_waypoint for x in runtime.guidance.agent_assignments]))
                    if mode in ('baseline','repeat'):signatures.append(signature(runtime,row))
                    else:
                        if row['safety_features'].shape!=(4,192) or len(row['safety_rows'])!=4:raise ValueError('all-role safety contract failed')
                        actual=runtime.env.unwrapped._agent_acc.detach().cpu().numpy()
                        if not np.isfinite(actual).all():raise ValueError('nonfinite physical control')
                summary=runtime.summary();write(directory/'episode.json',summary);write(directory/'physical_frames.json',frames)
                if mode=='baseline':
                    baseline=frames;baseline_signatures=signatures;write(directory/'signatures.json',signatures)
                    previous=ROOT/'verification/safe_rl_v1/238e7dd11199367e8275b1f5543e177dee5fe24e8ae635f9a56d1deb24c37d97'
                    metadata=json.loads((previous/'regression/checks.json').read_text(encoding='utf-8'))
                    report=archived_report(signatures,frames,previous/'native_evidence.zip',
                        dict(platform=metadata['platform'],python=metadata['python'],library_versions='not recorded'))
                    write(directory/'archived_reference_comparison.json',report)
                    result['archived_reference_comparison']=report
                if mode=='repeat':
                    write(directory/'signatures.json',signatures)
                    report=require_same_host_repeat(baseline_signatures,signatures,physical_steps=len(baseline))
                    write(directory/'same_host_parity.json',report);result['same_host_parity']=report
                if mode=='collection':
                    errors={key:max(float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key])))) for a,b in zip(baseline,frames)) for key in ('positions','velocities','rewards','tracking')}
                    if len(frames)!=len(baseline) or max(errors.values())>5e-5:raise ValueError('absolute input composition parity failed: '+str(errors))
                    write(directory/'composition_parity.json',dict(passed=True,max_errors=errors,scope='32-step numerical physical/reward/guidance parity; diagnostic prior/residual decomposition differs by explicit contract'))
                if mode not in ('baseline','repeat','collection'):
                    if summary['safety_rl']['counts'].get('agent_checks')!=4*runtime.step:raise ValueError('per-physical-step all-role checks incomplete')
                if mode=='learned' and learner.updates!=4:raise ValueError('temporary model exceeded four updates')
                records.append(dict(case=name,mode=mode,physical_steps=runtime.step,termination_reason=summary['termination_reason'],success=summary['success'],
                    collision_agents=summary.get('first_collision_agent_ids',[]),safety_counts=summary.get('safety_rl',{}).get('counts',{})))
            finally:
                if runtime is not None:runtime.close()
        require_review(source);result.update(passed=True,records=records,speed_audit=speed_audit(),baseline_same_host_repeat_exact=True,
            parent_cross_host_identity_claimed=False,composition_numerical_parity=True)
    except Exception:result['traceback']=traceback.format_exc()
    result.update(environment_initializations=starts,physical_steps=steps,additional_physical_steps=steps)
    if starts>8 or steps>256:result.update(passed=False,budget_violation=True)
    write(output/'safe_rl_workflow.json',result);return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    result=run(a.output) if a.execute else dict(executed=False,output=a.output)
    print(json.dumps(result,indent=2))
    if a.execute and not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
