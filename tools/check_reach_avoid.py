"""Eight bounded mechanism cases, never thesis training or checkpoint restore."""
import argparse,copy,hashlib,json,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review,training_admission
from chapter3_bser.learning.config import validate as validate_task
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.reach_avoid.runtime import ReachAvoidRuntime
from chapter3_bser.reach_avoid.options import validate
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from tools.safe_rl_acceptance import near_buffer_scene
from tools.safe_rl_validation import environment
from tools.audit_safe_speed import run as speed_audit


CASES=(('baseline_0002',1,'baseline'),('composition_0002',1,'nominal'),
       ('rule_0002',1,'rule'),('learned_wiring_0002',1,'learned'),
       ('fault_all_four',1,'fault'),('buffer_all_four',1,'buffer'),
       ('moving_handoff',1,'handoff'),('flow_equilibrium',5,'flow'))

def select_cases(names=None):
    if names is None:return CASES
    if not names or len(set(names))!=len(names) or set(names)-{x[0] for x in CASES}:raise ValueError('unique named bounded cases required')
    if 'composition_0002' in names and 'baseline_0002' not in names:raise ValueError('composition requires its current baseline')
    return tuple(x for x in CASES if x[0] in names)


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def custom_scene(parent,mode):
    if mode=='buffer':return near_buffer_scene(parent)
    scene=copy.deepcopy(parent)
    if mode not in ('handoff','flow'):return scene
    positions=([[3.,3.,4.5],[3.,9.,4.5],[9.,3.,4.5],[5.8,3.5,4.5]] if mode=='handoff'
               else [[3.,3.,4.5],[3.,9.,4.5],[9.,3.,4.5],[12.7296180725,18.7451057434,4.7090797424]])
    target=[5.,3.,4.5] if mode=='handoff' else [19.,19.,7.]
    obstacles=[dict(center=[17.,16.,4.5],size=[1.,1.,2.])]
    scene.update(scenario_id='reach_avoid_'+mode+'_mechanism_v1',initial_agent_positions=positions,
        initial_executor_wait_point=[13.,19.,5.5],target_position=target,target_initial_position=target,
        target_initial_velocity=[.15,.05,0.] if mode=='handoff' else [.1,0.,0.],
        obstacles=obstacles,obstacle_count=1,use_obstacles=True,
        obstacle_layout_sha256=hashlib.sha256(json.dumps(obstacles,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
    for key in list(scene):
        if key.startswith('target_trajectory') or 'reachability' in key:scene.pop(key)
    return scene


def require_case(summary,frames,initial,mode):
    steps=len(frames);trace=summary['reach_avoid']['trace']
    if not 1<=steps<=32 or summary.get('collision_records') or summary.get('first_collision_agent_ids'):
        raise ValueError('curated reach-avoid mechanism collided or exceeded its budget')
    if len(trace)!=4*steps or {(r['step'],r['agent_id']) for r in trace}!={(t,i) for t in range(steps) for i in range(4)}:
        raise ValueError('incomplete all-role physical trace')
    positions=np.array([initial,*[f['positions'] for f in frames]],float)
    if positions.shape!=(steps+1,4,3) or not np.isfinite(positions).all():raise ValueError('invalid mechanism frames')
    for row in trace:
        for key in ('executed_action','actual_action_after_physics_mask'):
            if not np.isfinite(row[key]).all() or np.any(np.abs(row[key])>np.array([1.,1.,1.])+1e-5):raise ValueError('illegal physical command')
        if row['action_mode'].startswith('UNCERTIFIED') and row['model_checked']:raise ValueError('emergency mislabeled as checked')
    travel=np.linalg.norm(np.diff(positions,axis=0),axis=2).sum(0)
    displacement=np.linalg.norm(positions[-1]-positions[0],axis=1)
    if mode in ('fault','buffer'):
        if np.any(travel<.15) or np.any(displacement<.1):raise ValueError('curated recovery stopped instead of making progress')
        for i in range(4):
            if not any(r['agent_id']==i and r['selected_index']!=0 for r in trace):raise ValueError('fault did not exercise goal repair for every role')
    if mode=='buffer':
        for i in range(4):
            if positions[1,i,0]>=positions[0,i,0]-1e-4 or positions[0,i,0]-positions[-1,i,0]<.52:
                raise ValueError('near-buffer mechanism failed to exit margin')
            if not any(r['agent_id']==i and r['action_mode']=='BUFFER_EXIT' and r['model_checked'] for r in trace):
                raise ValueError('buffer case did not exercise checked margin exit')
    if mode=='handoff':
        if not summary.get('found') or summary.get('executor_target_received_step') is None:
            raise ValueError('public target handoff fixture was not exercised')
        if not any(r['agent_id']==3 and r['phase']==2 for r in trace):raise ValueError('post-handoff goal branch absent')
    if mode=='flow':
        goal=np.array([13.,19.,5.5]);before=np.linalg.norm(positions[0,3]-goal);after=np.linalg.norm(positions[-1,3]-goal)
        if not after<before-.1:raise ValueError('shared rule baseline did not escape recorded flow equilibrium')
    if mode=='learned' and not any(r['candidate_values'] is not None for r in trace if r['phase']!=3):
        raise ValueError('learned critic was not queried')
    return dict(passed=True,collision_free=True,physical_steps=steps,path_length=travel.tolist(),net_displacement=displacement.tolist(),
        original_completion_protocol=True,scope='Curated <=32-step mechanisms; not full-episode effectiveness or a safety guarantee')


def run(output,*,cases=None):
    planned=select_cases(cases)
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):raise ValueError('bounded evidence must stay under work')
    output.mkdir(parents=True,exist_ok=False)
    source=verify_reviewed_changes()
    from tools.reach_avoid_source import verify
    protection=verify(ROOT);options=validate();torch.set_num_threads(1)
    generated=build_scenario_manifests(count=12,generator_seed=827699,split='validation',profiles=['M20_MOVING_UNKNOWN_MULTI'])['M20_MOVING_UNKNOWN_MULTI']['scenarios']
    result=dict(passed=False,source=source,planned_environment_initializations=len(planned),max_episode_steps=32,
        checkpoint_file_restore_executed=False,max_complete_updates_per_model=4,task_training=False,research_training=False,
        synthetic_auxiliary_learning_only=True,full_82_96_step_regression=False,safety_guarantee=False,performance_passed=False,
        environment=environment(),parent_algorithm_protection=protection)
    starts=steps=0;records=[];baseline=None
    try:
        for name,index,mode in planned:
            require_review(source);scene=custom_scene(generated[index],mode)
            scene.update(max_steps=32,scenario_role='eval',scenario_split='eval')
            scene['scenario_id']='eval_gen827699__'+scene['scenario_id']
            config=json.loads((ROOT/'configs/b2_mechanism.json').read_text())
            config.update(max_steps=32,total_main_trajectories=1,max_total_environment_steps=32,
                output_dir=str(output/'unused_task_training'))
            config=validate_task(config);training_admission(mechanism=True,config=config)
            directory=output/name;directory.mkdir();learner=None;local_options=options
            if mode=='learned':
                # Synthetic tensors, four complete auxiliary updates, no model
                # file load and no research-episode updates. Never performance.
                from checks.reach_avoid_fixtures import learning_fixture
                learner,local_options,_=learning_fixture()
                metrics=[learner.update() for _ in range(4)]
                if not learner.ready or learner.updates!=4:raise ValueError('synthetic wiring learner not exercised')
                write(directory/'synthetic_updates.json',dict(metrics=metrics,not_research_training=True,complete_updates=4))
            faults=({i:tuple(box['center']) for i,box in enumerate(scene['obstacles'])} if mode=='buffer'
                    else {i:tuple(scene['obstacles'][0]['center']) for i in range(4)} if mode=='fault'
                    else {3:(13.,19.,5.5)} if mode=='flow' else None)
            starts+=1;write(output/f'initialization_{starts:02d}.json',dict(case=name,max_steps=32,source=source))
            write(directory/'scene.json',scene)
            print(f'[reach-avoid mechanism {starts}/8] {name}',flush=True)
            runtime=None;frames=[];before_case_steps=steps
            try:
                if mode=='baseline':runtime=RepairedMissionRuntime(config,scene,seed=int(scene['scenario_seed']))
                else:runtime=ReachAvoidRuntime(config,scene,seed=int(scene['scenario_seed']),options=local_options,learner=learner,
                    behavior=mode if mode in ('nominal','learned') else 'rule',goal_faults=faults)
                initial=runtime.env.get_agent_state().positions
                while not runtime.terminal:
                    row=runtime.advance(ZeroResidualPolicy(),explore=False);steps+=1
                    public=runtime.env.get_agent_state()
                    frames.append(dict(t=row['t'],positions=public.positions,velocities=public.velocities,
                        rewards=row['rewards'].tolist(),tracking=[x.tracking_waypoint for x in runtime.guidance.agent_assignments]))
                    if runtime.step%8==0:print(f'[reach-avoid] {name}: {runtime.step}/32 physical steps',flush=True)
                summary=runtime.summary();write(directory/'episode.json',summary);write(directory/'physical_frames.json',frames)
                if mode=='baseline':baseline=frames
                elif mode=='nominal':
                    if len(frames)!=len(baseline):raise ValueError('nominal composition length changed')
                    errors={key:max(float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key])))) for a,b in zip(baseline,frames))
                            for key in ('positions','velocities','rewards','tracking')}
                    if max(errors.values())>5e-5:raise ValueError('public snapshot composition changed original behavior: '+str(errors))
                    result['composition_parity']=dict(passed=True,max_errors=errors,tolerance=5e-5,scope='same-host32-step physical/reward/tracking prefix')
                    write(directory/'composition_parity.json',result['composition_parity'])
                else:
                    acceptance=require_case(summary,frames,initial,mode);write(directory/'acceptance.json',acceptance)
                records.append(dict(case=name,mode=mode,physical_steps=runtime.step,termination_reason=summary['termination_reason'],
                    success=summary['success'],collision_agents=summary.get('first_collision_agent_ids',[]),evidence_origin='current_execution',source_identity=source))
                print(f'[reach-avoid] {name}: completed',flush=True)
            finally:
                if runtime is not None:
                    steps=max(steps,before_case_steps+runtime.step)
                    write(directory/'physical_progress.json',dict(actual_physical_steps=runtime.step,source_identity=source))
                    runtime.close()
        require_review(source);result.update(passed=True,records=records,speed_audit=speed_audit())
    except Exception:result['traceback']=traceback.format_exc()
    result.update(records=records,environment_initializations=starts,physical_steps=steps,
        additional_environment_initializations=starts,additional_physical_steps=steps,
        current_source_environment_initializations=starts,current_source_physical_steps=steps)
    if starts>8 or steps>256:result.update(passed=False,budget_violation=True)
    write(output/'reach_avoid_workflow.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--cases',nargs='+');p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    result=run(a.output,cases=a.cases) if a.execute else dict(executed=False,output=a.output,cases=a.cases)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if a.execute and not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
