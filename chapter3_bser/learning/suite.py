"""Prepared, manual D2/B2/B3 runs with immutable inputs and final checkpoints."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import uuid
from project_paths import ROOT
from chapter3_bser.learning.checkpoint import digest, write_json, state_digest
from chapter3_bser.learning.config import validate
from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review
from chapter3_bser.learning.scenarios import validate_manifest
from chapter3_bser.learning.evaluation_records import verified_rows

ARMS = ('D2', 'D2_B2', 'D2_B3')
METHODS = {'D2': 'D2', 'D2_B2': 'D2_B2_repair_v2', 'D2_B3': 'D2_B3_repair_v2'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, manifest, *, arms=ARMS, seeds=(2729, 3729, 4729), mechanism=False, b2_config=None, b3_config=None):
    identity = verify_reviewed_changes()
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    if 'collision_terminal' not in root.parts:
        raise ValueError('suite root must contain collision_terminal')
    arms, seeds = list(arms), list(seeds)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS) or not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('unique supported arms and nonnegative training seeds required')
    suffix = '_mechanism' if mechanism else ''
    templates = {'D2_B2': validate(read(b2_config or ROOT / f'configs/b2{suffix}.json')),
                 'D2_B3': validate(read(b3_config or ROOT / f'configs/b3{suffix}.json'))}
    scenes = read(manifest)
    for template in templates.values():
        validate_manifest(scenes, template, evaluation=True, mechanism=mechanism)
    # Environment/task conditions must agree even when one arm is selected.
    condition_keys = ('profile', 'max_steps', 'reward', 'execution_runtime', 'task_protocol', 'collision_detection_revision',
                      'terminal_reward_revision', 'collision_terminal_reward', 'reward_objective', 'source_reward_revision',
                      'planner_protocol', 'repairs', 'waypoint_reward_timing')
    if any(templates['D2_B2'].get(k) != templates['D2_B3'].get(k) for k in condition_keys):
        raise ValueError('suite task/reward/planner conditions differ')
    jobs = []
    for arm in arms:
        for seed in ([None] if arm == 'D2' else seeds):
            name = arm if seed is None else f'{arm}_seed{seed}'
            config = copy.deepcopy(templates['D2_B2' if arm == 'D2' else arm])
            if seed is not None:
                config['seed'] = seed
            config['output_dir'] = str(root / 'jobs' / name / 'train')
            jobs.append(dict(name=name, arm=arm, method=METHODS[arm], training_seed=seed,
                             config=f'inputs/{name}.json', train=f'jobs/{name}/train', evaluation=f'jobs/{name}/evaluation', config_value=config))
    require_review(identity)
    write_json(root / 'inputs/evaluation.json', scenes)
    assets = {'inputs/evaluation.json': sha(root / 'inputs/evaluation.json')}
    for job in jobs:
        write_json(root / job['config'], job.pop('config_value'))
        assets[job['config']] = sha(root / job['config'])
    plan = dict(schema='auv.active.suite.v1', source_review=identity, mechanism_only=mechanism, jobs=jobs, assets=assets,
                training_protocol='current repair v2 native episode sampler unless scenario_manifest is explicitly configured',
                checkpoint_selection='final completed episode; no test-set selection',
                evaluation_pairing='same ordered physical scenarios and scenario_seed innovations for every policy',
                budget_policy='parent repair v2 finish-started-episode behavior', performance_passed=False)
    plan['sha256'] = digest(plan)
    require_review(identity)
    write_json(root / 'plan.json', plan)
    return plan


def load_plan(root):
    root = Path(root).resolve(); plan = read(root / 'plan.json')
    if plan.get('schema') != 'auv.active.suite.v1' or plan.get('sha256') != digest({k: v for k, v in plan.items() if k != 'sha256'}):
        raise ValueError('suite plan identity mismatch')
    require_review(plan['source_review'])
    for name, expected in plan['assets'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root / 'inputs') or sha(path) != expected:
            raise ValueError('prepared suite input changed: ' + name)
    return plan


@contextmanager
def run_lock(root):
    path = Path(root) / '.active_suite.lock'
    owner = json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), token=uuid.uuid4().hex))
    with path.open('x', encoding='utf-8') as stream:
        stream.write(owner)
    try:
        yield
    finally:
        if path.exists() and path.read_text(encoding='utf-8') == owner:
            path.unlink()


def training_receipt(root, plan, job):
    path = root / 'jobs' / job['name'] / 'training_receipt.json'
    if not path.exists():
        return None
    value = read(path)
    checkpoint = root / job['train'] / 'checkpoint_final.pt'
    if (value.get('source_review') != plan['source_review'] or value.get('plan_sha256') != plan['sha256']
            or value.get('config_sha256') != plan['assets'][job['config']] or value.get('checkpoint_sha256') != sha(checkpoint)
            or value.get('summary_sha256') != sha(root / job['train'] / 'summary.json')):
        raise ValueError('training completion receipt changed: ' + job['name'])
    return value


def run(root, *, stage='all', arms=None):
    root = Path(root).resolve(); plan = load_plan(root)
    if stage not in ('train', 'evaluate', 'all'):
        raise ValueError('invalid suite stage')
    selected = set(arms or [job['arm'] for job in plan['jobs']])
    if selected - {job['arm'] for job in plan['jobs']}:
        raise ValueError('selected arm absent from plan')
    result = []
    with run_lock(root):
        for phase in (('train', 'evaluate') if stage == 'all' else (stage,)):
            for job in plan['jobs']:
                if job['arm'] not in selected or phase == 'train' and job['arm'] == 'D2':
                    continue
                load_plan(root)
                config = read(root / job['config'])
                if phase == 'train':
                    receipt = training_receipt(root, plan, job)
                    if receipt is None:
                        from chapter3_bser.learning.train import Trainer
                        trainer = Trainer(config, root / job['train'], mechanism=plan['mechanism_only'])
                        summary = trainer.run()
                        receipt = dict(source_review=plan['source_review'], plan_sha256=plan['sha256'], config_sha256=plan['assets'][job['config']],
                                       checkpoint_sha256=sha(summary['latest_checkpoint']), summary_sha256=sha(root / job['train'] / 'summary.json'),
                                       model_state_sha256=state_digest(trainer.model.training_state_dict()))
                        require_review(plan['source_review'])
                        write_json(root / 'jobs' / job['name'] / 'training_receipt.json', receipt)
                else:
                    output = root / job['evaluation']
                    expected_policy = None
                    if job['arm'] != 'D2':
                        trained = training_receipt(root, plan, job)
                        if trained is None:
                            raise ValueError('verified final training checkpoint required: ' + job['name'])
                        expected_policy = trained['model_state_sha256']
                    if output.exists():
                        _, receipt = verified_rows(output / 'episodes.jsonl', plan['source_review'])
                        if (receipt['manifest_sha256'] != digest(read(root / 'inputs/evaluation.json'))
                                or receipt['method'] != job['method'] or receipt['mechanism_only'] != plan['mechanism_only']
                                or expected_policy is not None and receipt['policy_id'] != expected_policy):
                            raise ValueError('existing evaluation differs from this plan')
                    elif job['arm'] == 'D2':
                        from chapter3_bser.learning.reference import evaluate_reference
                        evaluate_reference(config, root / 'inputs/evaluation.json', output, mechanism=plan['mechanism_only'])
                    else:
                        from chapter3_bser.learning.evaluate import evaluate
                        evaluate(root / job['train'] / 'checkpoint_final.pt', root / 'inputs/evaluation.json', output, mechanism=plan['mechanism_only'])
                result.append(dict(job=job['name'], stage=phase, status='completed'))
    require_review(plan['source_review'])
    return dict(executed=True, jobs=result, performance_passed=False)


def summarize_suite(root, output):
    root = Path(root).resolve(); plan = load_plan(root)
    paths, expected, missing = [], {}, []
    for job in plan['jobs']:
        expected[job['method']] = expected.get(job['method'], 0) + 1
        path = root / job['evaluation'] / 'episodes.jsonl'
        if (path.parent / 'evaluation_identity.json').is_file():
            paths.append(path)
        else:
            missing.append(job['name'])
    from chapter3_bser.learning.summarize import summarize
    result = summarize(paths, output, expected=expected)
    result['missing_jobs'] = missing
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--root', required=True); prep.add_argument('--manifest', required=True)
    prep.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS)); prep.add_argument('--seeds', nargs='+', type=int, default=[2729, 3729, 4729])
    prep.add_argument('--b2-config'); prep.add_argument('--b3-config'); prep.add_argument('--mechanism-check', action='store_true')
    launch = commands.add_parser('run')
    launch.add_argument('--root', required=True); launch.add_argument('--stage', choices=('train', 'evaluate', 'all'), default='all')
    launch.add_argument('--arms', nargs='+', choices=ARMS)
    report = commands.add_parser('summarize')
    report.add_argument('--root', required=True); report.add_argument('--output', required=True)
    for sub in (prep, launch, report):sub.add_argument('--execute', action='store_true')
    args = vars(parser.parse_args(argv)); execute = args.pop('execute'); command = args.pop('command')
    if not execute:
        result = dict(executed=False, command=command, **args)
    elif command == 'prepare':
        args['mechanism'] = args.pop('mechanism_check'); result = prepare(**args)
    elif command == 'run':result = run(**args)
    else:result = summarize_suite(**args)
    print(json.dumps(result, indent=2)); return result
