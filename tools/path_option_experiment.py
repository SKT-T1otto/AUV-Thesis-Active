"""Frozen OLD path-option plans, fresh training, evaluation and honest reporting."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from project_paths import ROOT
from chapter3_bser.path_option.options import DEFAULTS, VARIANTS, INTERVENTIONS, validate
from chapter3_bser.path_option.source import identity, admission, write, sha, digest, new_output
from chapter3_bser.path_option.workflow import task_config, check_splits, train, evaluate, load_policy

PROFILE = 'M20_MOVING_UNKNOWN_MULTI'
CATEGORIES = ('invalid_goal', 'detour', 'new_observation', 'distant_goal', 'turn_braking')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def generate(count, seed, role):
    from core.scenarios.ch3_generator_impl import build_scenario_manifests
    m = build_scenario_manifests(count=count, generator_seed=seed,
        split='train' if role == 'train' else 'validation', profiles=[PROFILE])[PROFILE]
    for n, s in enumerate(m['scenarios']):
        s.update(scenario_id=f'po_{role}_{seed}_{n:04d}', scenario_role=role, scenario_split=role)
    return m


def special_manifest(count=20, seed=940001):
    """Frozen geometry creates challenges; logs determine whether they were encountered."""
    from core.scenarios.ch3_generator_impl import _unknown_trajectory_record, _validate_unknown_truth_reachability
    base = generate(1, seed, 'eval')['scenarios'][0]
    rng = np.random.default_rng(seed)
    scenes = []; faults = {}; labels = {}
    for group, category in enumerate(CATEGORIES):
        for n in range(count):
            s = copy.deepcopy(base)
            centers = [[10., 10., 4.]]
            sizes = [[1.2, 12., 7.8]]
            agents = np.array([[3., 4., 2.], [3., 8., 4.], [3., 12., 6.], [3., 16., 3.]])
            if category == 'detour':
                centers = [[8., 10., 4.], [5., 15.5, 4.], [5., 4.5, 4.]]
                sizes = [[1., 10., 7.8], [7., 1., 7.8], [7., 1., 7.8]]
                agents = np.array([[4., 10., 2.], [2., 2., 3.], [2., 18., 5.], [4., 10., 6.]])
            elif category == 'turn_braking':
                centers.append([13., 15.5, 4.]); sizes.append([6., 1., 7.8])
            agents += rng.uniform(-.12, .12, agents.shape)
            boxes = [dict(center=c, size=z) for c, z in zip(centers, sizes)]
            target = [17., 18., 5.]; wait = [17., 3., 3.]
            # Keep the original speed and reflection law, but move within the
            # public upper corridor. The old diagonal trajectory grazed the
            # wall at target clearance .2, inside the planner's .4 clearance,
            # so the full-trajectory feasibility gate correctly rejected it.
            velocity = [float(np.hypot(.30, .12)), 0., .03]
            number = group*count+n
            name = f'po_special_{category}_{seed}_{n:04d}'
            trajectory, distance, points = _unknown_trajectory_record(target, velocity, boxes)
            reach = _validate_unknown_truth_reachability(agents, wait, target, points, boxes)
            s.update(scenario_id=name, scenario_seed=seed+number, planner_seed=seed+number,
                initial_agent_positions=agents.tolist(), initial_executor_wait_point=wait,
                target_position=target, target_initial_position=target, target_initial_velocity=velocity,
                target_motion_seed=seed+number, obstacles=boxes, obstacle_count=len(boxes),
                obstacle_layout_id='custom_aabb_v1', obstacle_layout_sha256=digest(boxes),
                target_trajectory_sha256=trajectory, target_trajectory_distance=distance,
                obstacle_family='path_option_challenge_v2', **reach)
            if category == 'invalid_goal':
                agent = n % 4; point = centers[0]
            elif category == 'detour':
                agent = 0; point = [15., 10., 2.]
            else:
                agent = n % 4; point = [18., 4.+(n % 4)*4., 3.]
            faults[name] = [dict(agent=agent, point=point, start=0, end=400)]
            labels[name] = dict(category=category, affected_agent=agent,
                challenge_revision='path_option_challenge_v2',
                qualification='designed challenge; occurrence must be verified from trace',
                truth_used_only_for_offline_generation=True)
            scenes.append(s)
    return dict(scenarios=scenes, scenario_count=len(scenes)), faults, labels


def prepare(args):
    source = identity(); out = new_output(args.output)
    config = task_config(read(args.config))
    options = validate(read(args.options))
    if config['max_steps'] != 400:
        raise ValueError('research plan requires original 400-step episodes')
    manifests = {}
    for name, count, seed, role in (('training', 32, 910001, 'train'),
                                    ('development', 12, 920001, 'eval'),
                                    ('normal', 100, 930001, 'eval')):
        print(f'[path-option prepare] generating {name}: {count} scenes', flush=True)
        manifests[name] = generate(count, seed, role)
    print('[path-option prepare] generating special: 5 x 20 scenes', flush=True)
    special, faults, labels = special_manifest()
    manifests['special'] = special
    check_splits(manifests['training']['scenarios'], manifests['development']['scenarios'],
        manifests['normal']['scenarios']+special['scenarios'])
    out.mkdir(parents=True)
    for name, value in manifests.items():
        write(out/(name+'.json'), value)
    write(out/'faults.json', faults); write(out/'labels.json', labels)
    write(out/'config.json', config); write(out/'options.json', options)
    seeds = [20261111] if args.stage == 'pilot' else [20261111, 20261112, 20261113]
    arms = ['A', 'B', 'C'] if args.stage == 'pilot' else ['A', 'B', 'C', 'D']
    inputs = {p.name: sha(p) for p in out.glob('*.json')}
    plan = dict(schema='auv.path_option.plan.v1', source=source, stage=args.stage, inputs=inputs,
        primary_arms=arms, supplemental_arms=list(VARIANTS[4:]), seeds=seeds,
        episodes=100 if args.stage == 'pilot' else 500, eval_every=25,
        maximum_episode_steps=400, training_from_scratch=True, resume=False,
        selection_rule='fixed final-budget policy is primary; development selection is secondary only',
        bootstrap_episodes=0, pretrain_updates=0, safety_filter_common=True,
        independent_test_usage='run only after algorithm and settings are frozen')
    write(out/'plan.json', plan)
    return dict(prepared=True, root=str(out), plan=plan)


def frozen(root):
    root = Path(root).resolve(); plan = read(root/'plan.json')
    if plan['source'] != identity():
        raise ValueError('plan source changed')
    for name, expected in plan['inputs'].items():
        if sha(root/name) != expected:
            raise ValueError('frozen input changed: '+name)
    return root, plan


def run(args):
    admission(args.validation)
    root, plan = frozen(args.root)
    if args.seed not in plan['seeds']:
        raise ValueError('seed not in frozen stage')
    if args.arm not in plan['primary_arms'] and not args.supplemental:
        raise ValueError('supplemental arms require explicit --supplemental')
    output = root/'jobs'/f'{args.arm}_seed{args.seed}'
    output.parent.mkdir(exist_ok=True)
    lock = output.with_suffix('.lock')
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        return train(read(root/'config.json'), read(root/'training.json')['scenarios'],
            read(root/'development.json')['scenarios'], read(root/'options.json'), args.arm, output,
            episodes=plan['episodes'], eval_every=plan['eval_every'], seed=args.seed)
    except Exception:
        import traceback
        if output.exists() and not (output/'failure.json').exists():
            write(output/'failure.json', dict(source=identity(), traceback=traceback.format_exc(),
                resume_supported=False, partial_outputs_preserved=True))
        raise
    finally:
        lock.unlink()


def evaluation(args):
    source = admission(args.validation); root, plan = frozen(args.root)
    config = read(root/'config.json'); scenes = read(root/(args.suite+'.json'))['scenarios']
    learner = None
    if args.mode == 'learned':
        if not args.policy:
            raise ValueError('new policy required')
        learner = load_policy(args.policy, scenes, source, config_hash=digest(config))
    elif args.policy or args.intervention != 'none':
        raise ValueError('rule/native D2 baselines cannot load or intervene on weights')
    output = new_output(args.output)
    request = dict(source=source, mode=args.mode, suite=args.suite, intervention=args.intervention,
        expected_scenes=len(scenes), plan_sha256=sha(root/'plan.json'))
    if args.mode == 'd2':
        from chapter3_bser.learning.runtime import RepairedMissionRuntime
        from chapter3_bser.learning.reference import ZeroResidualPolicy
        from chapter3_bser.path_option.metrics import episode, aggregate
        from chapter3_bser.path_option.workflow import journal
        output.mkdir(parents=True); write(output/'evaluation_request.json', request); rows = []
        if args.suite != 'normal':
            raise ValueError('native D2 baseline uses normal suite; specialized faults require new runtime')
        for n, scene in enumerate(scenes):
            r = RepairedMissionRuntime(config, scene, seed=950001+n, map_research=None)
            try:
                while not r.terminal:
                    r.advance(ZeroResidualPolicy(), explore=False)
                raw = r.summary(); metric = episode(raw, 400)
                journal(output/'episodes.jsonl', raw); journal(output/'metrics.jsonl', metric); rows.append(metric)
            finally:
                r.close()
        summary = aggregate(rows); write(output/'summary.json', summary)
    else:
        summary = evaluate(config, scenes, read(root/'options.json') if learner is None else learner.o,
            learner, output, seed=950001, intervention=args.intervention,
            faults_by_id=read(root/'faults.json') if args.suite == 'special' else None, request=request)
    if args.suite == 'special':
        from tools.path_option_report import read_lines
        from chapter3_bser.path_option.metrics import aggregate
        labels = read(root/'labels.json'); rows = read_lines(output/'metrics.jsonl')
        by_category = {category: aggregate([r for r in rows if labels[r['scenario_id']]['category'] == category])
            for category in CATEGORIES}
        write(output/'categories.json', dict(results=by_category,
            qualification='Designed challenges; use raw trace to verify actual mechanism occurrence'))
    write(output/'evaluation.json', dict(source=identity(), plan_sha256=sha(root/'plan.json'),
        mode=args.mode, suite=args.suite, intervention=args.intervention,
        training_seed=None if learner is None else learner.o['seed'], evaluation_seed=950001,
        retrained_variant=None if learner is None else learner.variant,
        inference_intervention=args.intervention != 'none',
        policy_sha256=None if learner is None else sha(args.policy)))
    return summary


def report(args):
    from tools.path_option_report import build_report
    root, plan = frozen(args.root)
    return build_report(root, new_output(args.output), plan)


def status(args):
    from tools.path_option_report import read_lines
    root, plan = frozen(args.root)
    rows = []
    for arm in plan['primary_arms']:
        for seed in plan['seeds']:
            folder = root/'jobs'/f'{arm}_seed{seed}'
            recent = read_lines(folder/'training.jsonl')
            rows.append(dict(arm=arm, seed=seed, completed=(folder/'completed.json').exists(),
                failed=(folder/'failure.json').exists(), latest=recent[-1] if recent else None,
                seconds_since_training_journal=None if not recent else time.time()-(folder/'training.jsonl').stat().st_mtime))
    return dict(jobs=rows, read_only=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--stage', choices=('pilot', 'formal'), default='pilot')
    q.add_argument('--config', default='configs/b2.json'); q.add_argument('--output', required=True)
    q.add_argument('--options', default='configs/path_option_v1.json')
    q = sub.add_parser('run'); q.add_argument('--root', required=True); q.add_argument('--validation', required=True)
    q.add_argument('--arm', choices=VARIANTS, required=True); q.add_argument('--seed', type=int, required=True)
    q.add_argument('--supplemental', action='store_true')
    q = sub.add_parser('evaluate'); q.add_argument('--root', required=True); q.add_argument('--validation', required=True)
    q.add_argument('--suite', choices=('normal', 'special'), default='normal')
    q.add_argument('--mode', choices=('learned', 'rule', 'd2'), default='learned'); q.add_argument('--policy')
    q.add_argument('--intervention', choices=INTERVENTIONS, default='none'); q.add_argument('--output', required=True)
    q = sub.add_parser('report'); q.add_argument('--root', required=True); q.add_argument('--output', required=True)
    q = sub.add_parser('status'); q.add_argument('--root', required=True)
    for q in sub.choices.values():
        q.add_argument('--execute', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.execute:
        value = dict(executed=False, requested=vars(args), note='No environment, output, policy restore or training executed')
    else:
        torch.set_num_threads(1)
        value = dict(prepare=prepare, run=run, evaluate=evaluation, report=report, status=status)[args.command](args)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return value


if __name__ == '__main__':
    main()
