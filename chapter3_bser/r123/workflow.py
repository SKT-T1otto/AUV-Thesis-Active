"""Explicit bounded-budget full-mission research, fixed final policies, fresh paths."""
import copy
import json
import random
import time
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.path_option.workflow import task_config, check_splits, journal
from chapter3_bser.path_option.metrics import episode, aggregate
from chapter3_bser.r123.options import SCHEMA, CONTROL, options
from chapter3_bser.r123.runtime import PathOptionRuntime
from chapter3_bser.r123.learner import Learner
from chapter3_bser.r123.source import identity, write, sha, digest


@contextmanager
def preserve_rng(learner):
    py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state()
    lr=None if learner is None else copy.deepcopy(learner.rng.bit_generator.state)
    try:yield
    finally:
        random.setstate(py);np.random.set_state(nr);torch.set_rng_state(tr)
        if learner is not None:learner.rng.bit_generator.state=lr


def metric(raw, maximum):
    m=episode(raw,maximum);events=raw['path_option']['lifecycle_events']
    trace=[x for x in raw['path_option']['trace'] if x['phase']!=3]
    m.update(task_accepts=sum(e['event']=='TASK_ACCEPT' for e in events),
        task_releases=sum(e['event']=='TASK_RELEASE' for e in events),
        recovery_returns=sum(e['event']=='RECOVERY_RETURN' for e in events),
        capture_contact_steps=raw.get('capture_contact_step_count',0),
        capture_hold_max=raw.get('capture_hold_counter_max',0),
        handoff=int(raw.get('executor_target_received_step') is not None),
        motion_distance=sum(float(np.linalg.norm(np.array(t['next_position'])-t['position'])) for t in trace),
        probe_fraction=sum(t['decision']['action']>=10 for t in trace)/max(1,len(trace)))
    return m


def one_episode(config,scene,opts,learner,*,arm,seed,episode_id,training=False,
                remaining_steps=None,update_callback=None,step_hook=None):
    runtime=PathOptionRuntime(config,scene,seed=seed,episode_id=episode_id,options=opts,learner=learner,arm=arm)
    consumed=0;steps=0;start=time.monotonic()
    try:
        while not runtime.terminal and (remaining_steps is None or steps<remaining_steps):
            runtime.advance(ZeroResidualPolicy(),explore=training);steps+=1
            if step_hook is not None:step_hook(steps)
            if training:
                for t in runtime.transitions[consumed:]:learner.add(t)
                consumed=len(runtime.transitions)
                update_callback(learner)
            if steps%25==0:
                print(f'[r123 {arm} {"train" if training else "eval"}] case={episode_id} step={steps}/{config["max_steps"]} updates={0 if learner is None else learner.updates}',flush=True)
        raw=runtime.summary()
        complete=runtime.terminal
        # A sampler budget boundary is NOT an environment terminal, failure,
        # collision or time-limit truncation; replay dones remain untouched.
        m=metric(raw,config['max_steps']) if complete else dict(scenario_id=scene['scenario_id'],steps=steps,budget_censored=True)
        m.update(wall_seconds=time.monotonic()-start)
        return raw,m,complete
    finally:runtime.close()


def evaluate(config,scenes,opts,learner,output,*,arm,seed=982001):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);metrics=[]
    before=None if learner is None else learning_state(learner)
    with preserve_rng(learner):
        for n,scene in enumerate(scenes):
            if learner is not None:learner.rng=np.random.default_rng(seed+n)
            raw,m,complete=one_episode(config,scene,opts,learner,arm=arm,seed=seed+n,episode_id=n)
            if not complete:raise ValueError('evaluation budget must complete each mission')
            journal(out/'episodes.jsonl',raw);journal(out/'metrics.jsonl',m);metrics.append(m)
    if learner is not None and before!=learning_state(learner):
        raise ValueError('evaluation mutated learner or optimizer')
    summary=aggregate(metrics)
    write(out/'summary.json',summary)
    write(out/'evaluation.json',dict(source=identity(),arm=arm,seed=None if learner is None else learner.o['seed'],
        config_hash=digest(config),control=CONTROL,options=opts,
        scenario_ids=[s['scenario_id'] for s in scenes],scenario_fingerprints=[scenario_fingerprint(s) for s in scenes],
        factual_full_episodes=True,learning_state_sha256=before,summary_sha256=sha(out/'summary.json')))
    return summary


def sha_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def learning_state(learner):
    def encode(value):
        if isinstance(value,torch.Tensor):return dict(shape=list(value.shape),sha256=sha_tensor(value))
        if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [encode(v) for v in value]
        return value
    return digest(encode(dict(policy=learner.policy.state_dict(),critics=learner.critics.state_dict(),
        target_policy=learner.target_policy.state_dict(),target_critics=learner.target_critics.state_dict(),
        actor_optimizer=learner.actor_optimizer.state_dict(),critic_optimizer=learner.critic_optimizer.state_dict(),
        updates=learner.updates,replay_size=len(learner.replay))))


def save_policy(path,learner,config,scenes):
    torch.save(learner.weights(),path)
    write(str(path)+'.json',dict(schema=SCHEMA,source=identity(),arm=learner.arm,options=learner.o,control=CONTROL,
        policy_sha256=sha(path),config_hash=digest(config),train_ids=[s['scenario_id'] for s in scenes],
        train_fingerprints=[scenario_fingerprint(s) for s in scenes],resume_supported=False))


def load_policy(path,config,scenes):
    m=json.loads(Path(str(path)+'.json').read_text(encoding='utf-8'))
    if (m.get('schema')!=SCHEMA or m['source']!=identity() or m['control']!=CONTROL
            or m['config_hash']!=digest(config) or m['policy_sha256']!=sha(path)):
        raise ValueError('R123 weight/source/config mismatch; old weights cannot be relabeled')
    if set(m['train_ids'])&{s['scenario_id'] for s in scenes} or set(m['train_fingerprints'])&{scenario_fingerprint(s) for s in scenes}:
        raise ValueError('train/evaluation leakage')
    model=Learner(m['options'],m['arm']);model.load_evaluation(torch.load(path,map_location='cpu',weights_only=True))
    return model


def train(config,scenes,dev,opts,arm,output,*,seed,step_budget,max_episodes=None):
    config=task_config(config);opts=options(opts);opts['seed']=seed
    check_splits(scenes,dev,[]);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    learner=Learner(opts,arm);start=time.monotonic();physical=0;ep=0
    write(out/'run.json',dict(schema=SCHEMA,source=identity(),arm=arm,switches=learner.switches,options=opts,
        control=CONTROL,step_budget=step_budget,max_episodes=max_episodes,from_scratch=True,resume=False,
        reward_changed=False,dynamics_changed=False,task_success_changed=False))
    def update(model):
        nonlocal physical
        physical+=1
        if physical%opts['update_every']==0:
            value=model.update()
            if value is not None:journal(out/'updates.jsonl',value)
    # Two explicitly named development evaluations, no hidden25-episode loops.
    initial=evaluate(config,dev,opts,learner,out/'development_initial',arm=arm)
    save_policy(out/'policy_initial.pt',learner,config,scenes)
    while physical<step_budget and (max_episodes is None or ep<max_episodes):
        ep+=1;scene=scenes[(ep-1)%len(scenes)]
        raw,m,complete=one_episode(config,scene,opts,learner,arm=arm,seed=seed+ep,episode_id=ep,
            training=True,remaining_steps=step_budget-physical,update_callback=update)
        journal(out/'training.jsonl',dict(episode=ep,**m,complete_episode=complete,physical_steps=physical,updates=learner.updates))
        journal(out/'training_trace.jsonl',dict(episode=ep,scenario_id=scene['scenario_id'],trace=raw['path_option']['trace'],
            lifecycle_events=raw['path_option']['lifecycle_events']))
        print(f'[r123 {arm}] train_steps={physical}/{step_budget} episodes={ep} elapsed={time.monotonic()-start:.1f}s',flush=True)
    final=evaluate(config,dev,opts,learner,out/'development_final',arm=arm)
    save_policy(out/'policy_final.pt',learner,config,scenes)
    passed=final['success']>initial['success'] and final['collision']<=initial['collision']
    write(out/'completed.json',dict(source=identity(),arm=arm,training_seed=seed,episodes=ep,physical_steps=physical,
        config_hash=digest(config),options_hash=digest({k:v for k,v in opts.items() if k!='seed'}),
        train_fingerprints=[scenario_fingerprint(s) for s in scenes],dev_fingerprints=[scenario_fingerprint(s) for s in dev],
        step_budget=step_budget,updates=learner.updates,initial=initial,final=final,
        screen_passed=bool(passed),performance_accepted=False,elapsed_seconds=time.monotonic()-start))
    return dict(output=str(out),physical_steps=physical,updates=learner.updates,screen_passed=bool(passed))
