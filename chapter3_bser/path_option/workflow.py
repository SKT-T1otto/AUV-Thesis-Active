"""Manual fresh training and frozen-policy evaluation. Never restores training."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.config import scenario_fingerprint, validate as validate_task
from chapter3_bser.path_option.options import SCHEMA, validate
from chapter3_bser.path_option.learner import Learner
from chapter3_bser.path_option.runtime import PathOptionRuntime
from chapter3_bser.path_option.source import identity, write, sha, digest
from chapter3_bser.path_option.metrics import episode, aggregate, learning_assessment, event_diagnostics


def journal(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def check_splits(train, development, test):
    previous_ids = set(); previous_physical = set()
    for scenes in (train, development, test):
        ids = [s['scenario_id'] for s in scenes]
        physical = [scenario_fingerprint(s) for s in scenes]
        if len(set(ids)) != len(ids) or len(set(physical)) != len(physical):
            raise ValueError('duplicate physical scene or ID')
        if previous_ids.intersection(ids) or previous_physical.intersection(physical):
            raise ValueError('physical training/development/test leakage')
        previous_ids.update(ids); previous_physical.update(physical)


def task_config(config):
    c = validate_task(config)
    m = c.get('map_experiment', {})
    if m.get('map_id', 'OLD') != 'OLD' or m.get('research_config') is not None or m.get('executor_safety', {}).get('enabled'):
        raise ValueError('OLD D2 only, without executor-specific guards')
    if c['max_steps'] > 400 or c['rl']['gamma'] != .95:
        raise ValueError('original time/discount contract required')
    return c


def one_episode(config, scene, options, learner, *, seed, episode_id, training=False,
                intervention='none', faults=None, update_hook=None, step_hook=None):
    runtime = PathOptionRuntime(config, scene, seed=seed, episode_id=episode_id, options=options,
        learner=learner, intervention=intervention, faults=faults)
    elapsed = time.monotonic(); consumed = 0; updates = []; steps = 0
    try:
        while not runtime.terminal:
            runtime.advance(ZeroResidualPolicy(), explore=training)
            steps += 1
            if training:
                for t in runtime.transitions[consumed:]:
                    learner.add(t)
                consumed = len(runtime.transitions)
                # One complete shared update per configured PHYSICAL step interval.
                if update_hook is not None:
                    update_hook(learner)
                elif steps % options['update_every'] == 0:
                    value = learner.update()
                    if value is not None:
                        updates.append(value)
            if step_hook is not None:
                step_hook(steps)
            if steps % 25 == 0:
                print(f'[path-option] episode={episode_id} step={steps}/{config["max_steps"]} updates={0 if learner is None else learner.updates}', flush=True)
        raw = runtime.summary()
        metric = episode(raw, config['max_steps'])
        metric.update(wall_seconds=time.monotonic()-elapsed)
        return raw, metric, updates
    finally:
        runtime.close()


def evaluate(config, scenes, options, learner, output, *, seed, intervention='none', faults_by_id=None, request=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if request is not None:
        write(output/'evaluation_request.json', request)
    metrics = []; diagnostics = []
    # Evaluation randomness must not advance the training policy RNG.
    saved = None if learner is None else copy.deepcopy(learner.rng.bit_generator.state)
    try:
        for index, scene in enumerate(scenes):
            if learner is not None:
                learner.rng = np.random.default_rng(seed+index)
            raw, metric, _ = one_episode(config, scene, options, learner, seed=seed+index,
                episode_id=index, intervention=intervention, faults=(faults_by_id or {}).get(scene['scenario_id']))
            journal(output/'episodes.jsonl', raw)
            journal(output/'metrics.jsonl', metric)
            metrics.append(metric)
            diagnostics.append(event_diagnostics(raw.get('path_option', {}).get('trace', [])))
    finally:
        if learner is not None:
            learner.rng.bit_generator.state = saved
    summary = aggregate(metrics)
    summary.update(event_diagnostics=diagnostics, intervention=intervention,
        inference_intervention=intervention != 'none', retrained_ablation=False)
    write(output/'summary.json', summary)
    return summary


def save_policy(path, learner, source, train_scenes, config_hash):
    torch.save(learner.weights(), path)
    write(str(path)+'.json', dict(schema=SCHEMA, source=source, options=learner.o,
        variant=learner.variant, policy_sha256=sha(path), config_sha256=config_hash,
        training_ids=[s['scenario_id'] for s in train_scenes],
        training_fingerprints=[scenario_fingerprint(s) for s in train_scenes],
        updates=learner.updates, training_resume_supported=False))


def load_policy(path, scenes, source, *, config_hash):
    metadata = json.loads(Path(str(path)+'.json').read_text(encoding='utf-8'))
    if (metadata['schema'] != SCHEMA or metadata['source'] != source or metadata['policy_sha256'] != sha(path)
            or metadata['config_sha256'] != config_hash or metadata['training_resume_supported'] is not False):
        raise ValueError('new algorithm/source/config/weight identity mismatch')
    if set(metadata['training_ids']) & {s['scenario_id'] for s in scenes} or set(metadata['training_fingerprints']) & {scenario_fingerprint(s) for s in scenes}:
        raise ValueError('training/evaluation leakage')
    learner = Learner(metadata['options'], metadata['variant'])
    learner.load_evaluation(torch.load(path, map_location='cpu', weights_only=True))
    return learner


def train(config, scenes, development, options, variant, output, *, episodes, eval_every, seed):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    config = task_config(config); options = validate(options); options['seed'] = seed
    check_splits(scenes, development, [])
    source = identity(); learner = Learner(options, variant)
    config_hash = digest(config)
    write(output/'run.json', dict(source=source, schema=SCHEMA, options=options, variant=variant,
        episodes=episodes, seed=seed, from_scratch=True, checkpoint_restore=False,
        reward_changed=False, dynamics_changed=False, phase_protocol_changed=False,
        bootstrap_episodes=0, pretrain_updates=0, replay='uniform_factual',
        hindsight=False, original_task_contract=[28, 3, 124]))
    start = time.monotonic(); physical = 0; curve = []; recent = []
    counters = dict(physical=0)

    def update_hook(model):
        counters['physical'] += 1
        if counters['physical'] % options['update_every'] == 0:
            update = model.update()
            if update is not None:
                journal(output/'updates.jsonl', update)

    def monitor(ep):
        weights = output/f'policy_{ep:06d}.pt'
        save_policy(weights, learner, source, scenes, config_hash)
        state = evaluate(config, development, options, learner, output/f'development_{ep:06d}', seed=920001)
        state.update(episode=ep, physical_steps=physical, updates=learner.updates, policy_sha256=sha(weights))
        curve.append(state); journal(output/'development.jsonl', state)
        return weights

    monitor(0)
    for ep in range(1, episodes+1):
        scene = scenes[(ep-1) % len(scenes)]
        raw, metric, _ = one_episode(config, scene, options, learner, seed=seed+ep,
            episode_id=ep, training=True, update_hook=update_hook)
        physical += metric['steps']; recent.append(metric)
        journal(output/'training.jsonl', dict(episode=ep, **metric, updates=learner.updates,
            elapsed_seconds=time.monotonic()-start,
            estimated_remaining_seconds=(time.monotonic()-start)/ep*(episodes-ep)))
        journal(output/'training_trace.jsonl', dict(episode=ep, scenario_id=scene['scenario_id'],
            trace=raw['path_option']['trace']))
        print(f'[path-option] completed={ep}/{episodes} steps={physical} updates={learner.updates} success={metric["success"]} elapsed={time.monotonic()-start:.1f}s', flush=True)
        if ep % eval_every == 0 or ep == episodes:
            final = monitor(ep)
    write(output/'completed.json', dict(source=identity(), episodes=episodes, physical_steps=physical,
        updates=learner.updates, final_policy=str(final.resolve()),
        assessment=learning_assessment(curve), performance_accepted=False, convergence_proved=False))
    return dict(output=str(output), physical_steps=physical, updates=learner.updates)
