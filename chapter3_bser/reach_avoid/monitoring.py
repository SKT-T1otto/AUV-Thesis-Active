"""Read-only training observer; explicit-research evaluation uses a policy copy.

Optimizer, replay, exploration schedule, dynamics and rewards stay unchanged.
Policy snapshots serve evaluation only and cannot resume training.
"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from chapter3_bser.learning.checkpoint import state_digest, digest
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.evaluate import preserved_rng
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.safe_rl.workflow import inputs
from chapter3_bser.reach_avoid.options import SCHEMA
from chapter3_bser.reach_avoid.learner import ReachAvoidLearner, first_event_cdf
from chapter3_bser.reach_avoid.goals import Goal
from chapter3_bser.reach_avoid.learning_diagnostics import settings, schedule, sha, create_json, episode_metrics, aggregate


def training_digest(learner):
    return state_digest(dict(policy=learner.state(),target_actor=learner.target_actor.state_dict(),
        target_critics=learner.target_critics.state_dict(),actor_optimizer=learner.actor_optimizer.state_dict(),
        critic_optimizer=learner.critic_optimizer.state_dict(),rng=copy.deepcopy(learner.rng.bit_generator.state),
        modes=[learner.actor.training,learner.critics.training,learner.target_actor.training,learner.target_critics.training]))


def parameters(learner):
    return [torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()
            for net in (learner.actor,learner.critics)]


def grad_norm(net):
    values=[p.grad.detach().double().square().sum() for p in net.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def one_step_scores(learner,records):
    """Score actual actions at h=1; later goal switches prevent naive h=32 labels."""
    queries=[];labels=[]
    for r in records:
        for i,raw in enumerate(r['goals']):
            if r['packet'].phases[i]==3:continue
            queries.append(np.concatenate((learner.inputs(r['packet'],i,Goal.from_array(raw)),r['actual_actions'][i])))
            collision=bool(r['team_collision'])
            labels.append([bool(r['reached'][i]) and not collision,collision])
    if not queries:return dict(count=0,brier_reach_h1=None,brier_collision_h1=None)
    with torch.no_grad():
        x=torch.tensor(np.asarray(queries),dtype=torch.float32)
        outputs=torch.stack([net(x) for net in learner.critics])
        cdf=first_event_cdf(outputs[:,:,:-1])
        reach=cdf[:,:,0,0].min(0).values.numpy();risk=cdf[:,:,0,1].max(0).values.numpy()
    y=np.asarray(labels,float)
    return dict(count=len(y),brier_reach_h1=float(np.mean((reach-y[:,0])**2)),
        brier_collision_h1=float(np.mean((risk-y[:,1])**2)),
        reach_positive_count=int(y[:,0].sum()),collision_positive_count=int(y[:,1].sum()),
        scope='One factual executed step; no32-step conditional-value calibration claim')


def evaluate_development(learner,state,config,scenes,options,spec,directory):
    """No file restore; called only by an explicitly admitted research run."""
    from chapter3_bser.reach_avoid.runtime import ReachAvoidRuntime
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    before=training_digest(learner);rows=[];scores=[]
    try:
        with preserved_rng(),torch.no_grad():
            policy=ReachAvoidLearner(options,ablation=learner.ablation)
            policy.load_for_evaluation(state)
            with (directory/'episodes.jsonl').open('x',encoding='utf-8') as stream:
                for index,scene in enumerate(scenes):
                    print(f'[development] {index+1}/{len(scenes)} scene={scene["scenario_id"]}',flush=True)
                    runtime=None
                    try:
                        runtime=ReachAvoidRuntime(config,copy.deepcopy(scene),seed=int(scene['scenario_seed']),
                            episode_id=index+1,options=options,learner=policy,behavior='learned',ablation=learner.ablation)
                        while not runtime.terminal:
                            runtime.advance(ZeroResidualPolicy(),explore=False)
                            if runtime.step%100==0:print(f'[development] case={index+1} step={runtime.step}/{config["max_steps"]}',flush=True)
                        summary=runtime.summary();row=episode_metrics(summary,config['max_steps'],spec)
                        row.update(scenario_id=scene['scenario_id'],exploration=False)
                        rows.append(row);scores.append(one_step_scores(policy,runtime.ra_records))
                        summary.update(development_only=True,exploration=False)
                        stream.write(json.dumps(summary,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    finally:
                        if runtime is not None:runtime.close()
        result=aggregate(rows,len(scenes));total=sum(s['count'] for s in scores)
        for key in ('brier_reach_h1','brier_collision_h1'):
            result[key]=sum(s[key]*s['count'] for s in scores if s['count'])/total if total else None
        result.update(event_score_count=total,exploration=False,optimizer_updates_during_evaluation=0,
            raw_episodes_sha256=sha(directory/'episodes.jsonl'),
            event_score_scope='Actual executed actions, one-step events; no32-step calibration claim')
        create_json(directory/'metrics.json',dict(result,episode_metrics=rows))
        return result
    except Exception as exc:
        create_json(directory/'failure.json',dict(error=repr(exc),complete_cases=len(rows),expected_cases=len(scenes)))
        raise
    finally:
        if training_digest(learner)!=before:
            raise RuntimeError('development evaluation changed training state/targets/optimizers/local RNG')


class TrainingMonitor:
    @classmethod
    def from_args(cls,args,output,source,config,scenes,options,training_ids,fingerprints):
        path=getattr(args,'dev_scenarios',None)
        if not path:return None
        spec=settings(json.loads(Path(args.monitor_settings).read_text(encoding='utf-8')))
        raw=json.loads(Path(path).read_text(encoding='utf-8'));count=len(raw.get('scenarios',[]))
        _,development=inputs(args.config,path,training=False,episodes=count)
        ids=set(training_ids)|{s['scenario_id'] for s in scenes}
        physical=set(fingerprints)|{scenario_fingerprint(s) for s in scenes}
        if ids&{s['scenario_id'] for s in development} or physical&{scenario_fingerprint(s) for s in development}:
            raise ValueError('development overlaps current/future training or bootstrap data')
        return cls(output,source,config,development,options,spec,args.episodes,sha(path),ids,physical)

    def __init__(self,output,source,config,development,options,spec,episodes,development_sha256,training_ids,fingerprints):
        self.root=Path(output);self.path=self.root/'monitoring';self.source=copy.deepcopy(source)
        self.config=copy.deepcopy(config);self.development=copy.deepcopy(development);self.options=copy.deepcopy(options)
        self.spec=settings(spec);self.episodes=episodes;self.points=schedule(episodes,self.spec['eval_every'])
        self.development_sha256=development_sha256;self.ids=set(training_ids);self.fingerprints=set(fingerprints)
        self.curve=[];self.logs={};self.last_update=0;self.last_episode=0;self.episode_before=None

    def start(self,learner):
        self.path.mkdir(parents=True,exist_ok=False)
        create_json(self.path/'protocol.json',dict(schema='auv.reach_avoid.monitoring.v1',source_identity=self.source,
            config_sha256=digest(self.config),options_sha256=digest(self.options),settings=self.spec,
            online_training_episodes=self.episodes,max_episode_steps=self.config['max_steps'],
            evaluation_points=self.points,development_sha256=self.development_sha256,development_cases=len(self.development),
            learning_seed=self.options['seed'],evaluation_exploration=False,
            rule_or_other_baseline_experiments=False,final_test_used=False,automatic_early_stop=False,
            episode0='Learned policy after offline pretraining, before online episode1',
            snapshot_purpose='Evaluation only; optimizer/replay not saved; training resume unsupported'))
        create_json(self.path/'development_scenarios.json',dict(scenarios=self.development))
        for key in ('updates','training_episodes','validation','snapshots'):
            self.logs[key]=(self.path/(key+'.jsonl')).open('x',encoding='utf-8')
        self.last_update=learner.updates

    def append(self,name,row):
        self.logs[name].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.logs[name].flush()

    def on_update(self,metric,learner,episode,environment_steps,phase):
        if metric is None:return
        row=dict(metric,phase=phase,episode=episode,training_environment_steps=environment_steps,
            actor_gradient_norm=grad_norm(learner.actor),critic_gradient_norm=grad_norm(learner.critics))
        if metric['complete_updates']!=learner.updates or learner.updates<=self.last_update:
            raise ValueError('nonmonotone real optimizer update counter')
        if any(not np.isfinite(float(row[k])) for k in ('actor_loss','event_loss_and_task_loss','dual','actor_gradient_norm','critic_gradient_norm')):
            create_json(self.path/'failure.json',dict(reason='nonfinite optimization metric',episode=episode,updates=learner.updates))
            raise FloatingPointError('nonfinite training metric; see monitoring/failure.json')
        self.append('updates',row);self.last_update=learner.updates

    def after_pretrain(self,learner):
        self.checkpoint(learner,0,0)

    def before_episode(self,learner):
        self.episode_before=parameters(learner);self.episode_start_updates=learner.updates

    def on_episode(self,learner,summary,episode,environment_steps,training_ids,fingerprints):
        self.ids.update(training_ids);self.fingerprints.update(fingerprints)
        after=parameters(learner)
        delta=[float(torch.linalg.vector_norm(b-a)) for a,b in zip(self.episode_before,after)]
        row=episode_metrics(summary,self.config['max_steps'],self.spec)
        row.update(episode=episode,training_environment_steps=environment_steps,updates=learner.updates,
            updates_this_episode=learner.updates-self.episode_start_updates,
            actor_parameter_delta_l2=delta[0],critic_parameter_delta_l2=delta[1])
        self.append('training_episodes',row);self.last_episode=episode;self.episode_before=None
        if episode in self.points:self.checkpoint(learner,episode,environment_steps)

    def checkpoint(self,learner,episode,environment_steps):
        if not learner.ready:raise ValueError('monitored policy is not genuinely evaluation-ready')
        state=learner.state();folder=self.path/'policies';folder.mkdir(exist_ok=True)
        path=folder/f'policy_episode_{episode:06d}.pt'
        with path.open('xb') as stream:torch.save(state,stream)
        metadata=dict(schema=SCHEMA,source_identity=self.source,sha256=sha(path),state_sha256=state_digest(state),
            ablation=learner.ablation,training_scenario_ids=sorted(self.ids),training_fingerprints=sorted(self.fingerprints),
            resume_supported=False,old_checkpoint_relabelled=False,performance_passed=False,
            training_episode=episode,training_environment_steps=environment_steps,updates=learner.updates,
            development_sha256=self.development_sha256,selection_scope='Development only, never final test')
        create_json(str(path)+'.json',metadata)
        self.append('snapshots',dict(episode=episode,path=path.relative_to(self.root).as_posix(),
            policy_sha256=metadata['sha256'],metadata_sha256=sha(str(path)+'.json'),state_sha256=metadata['state_sha256']))
        print(f'[monitor] evaluating learned snapshot episode={episode}, updates={learner.updates}',flush=True)
        metrics=evaluate_development(learner,state,self.config,self.development,self.options,self.spec,
            self.path/'evaluations'/f'episode_{episode:06d}')
        row=dict(metrics,episode=episode,training_environment_steps=environment_steps,updates=learner.updates,
            development_sha256=self.development_sha256,policy_path=path.relative_to(self.root).as_posix(),
            policy_sha256=metadata['sha256'],policy_state_sha256=metadata['state_sha256'],
            metrics_path=f'monitoring/evaluations/episode_{episode:06d}/metrics.json',
            metrics_sha256=sha(self.path/'evaluations'/f'episode_{episode:06d}'/'metrics.json'))
        self.append('validation',row);self.curve.append(row)
        from chapter3_bser.reach_avoid.monitor_reports import render
        with preserved_rng():report=render(self.root,self.path/'reports'/f'episode_{episode:06d}')
        print(f'[monitor] episode={episode} success={row["success_rate"]:.3f} collision={row["collision_rate"]:.3f} timeout={row["timeout_rate"]:.3f} status={report["assessment"]["status"]}',flush=True)

    def finish(self,learner,training_ids,fingerprints):
        if self.last_episode!=self.episodes or [r['episode'] for r in self.curve]!=self.points:
            raise ValueError('incomplete monitored experiment')
        self.ids.update(training_ids);self.fingerprints.update(fingerprints);self.close()
        best=max(self.curve,key=lambda r:(r['success_rate'],-r['collision_rate'],-r['timeout_rate'],
            -r['stagnation_rate'],r['team_discounted_return'],-r['episode']))
        create_json(self.path/'best_policy.json',dict(source_identity=self.source,
            training_episode=best['episode'],path=best['policy_path'],sha256=best['policy_sha256'],
            selection='Success descending, collision/timeout/stagnation ascending, original return descending, earlier tie',
            development_sha256=self.development_sha256,final_test_used=False,resume_supported=False))
        create_json(self.path/'completed.json',dict(source_identity=self.source,online_episodes=self.episodes,
            optimizer_updates=learner.updates,evaluation_points=self.points,
            development_environment_steps=sum(round(r['actual_length']*r['episodes']) for r in self.curve),
            journals={k:sha(self.path/(k+'.jsonl')) for k in self.logs},
            policy_snapshots_for_evaluation_only=True,research_performance_passed=False))
        from chapter3_bser.reach_avoid.monitor_reports import render
        with preserved_rng():render(self.root,self.path/'final_report')

    def close(self):
        for stream in self.logs.values():
            if not stream.closed:stream.close()
