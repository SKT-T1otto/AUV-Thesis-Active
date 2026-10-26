"""Run current regression and optional bounded rollout; never run thesis training."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest
from project_paths import ROOT


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def structural():
    from tools.verify_source import verify
    result = verify(ROOT)
    modules = []
    for path in sorted(ROOT.rglob('*.py')):
        if any(part in ('__pycache__', 'work', 'outputs') for part in path.relative_to(ROOT).parts):
            continue
        # Attached document builders are reference material, not importable
        # runtime/check code. Audit every declared Active Python source only.
        relative = path.relative_to(ROOT).as_posix()
        if relative.split('/')[0] not in ('core', 'chapter3_bser', 'checks', 'tools') and relative not in ('cli.py', 'project_paths.py'):
            continue
        source = path.read_text(encoding='utf-8'); tree = ast.parse(source)
        if relative.startswith(('core/', 'chapter3_bser/')):
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
                if any(n.split('.')[0] in ('tests', 'legacy_adapters', 'chapter4_rcag', 'chapter5_vsgc')
                       or ('.experiments.' in n and not (n == 'chapter3_bser.experiments.coupled_search_v1'
                           or n.startswith('chapter3_bser.experiments.coupled_search_v1.'))) for n in names):
                    raise ValueError('forbidden runtime import in ' + relative)
        module = relative[:-3].replace('/', '.')
        if module.endswith('.__init__'):module = module[:-9]
        imported = importlib.import_module(module)
        if not Path(imported.__file__).resolve().is_relative_to(ROOT):
            raise ValueError('module leaked outside rebuilt directory: ' + module)
        modules.append(module)
    return dict(passed=True, source=result, imported_modules=modules)


def regression(output):
    import tempfile
    expected = json.loads((ROOT/'configs/rebuild_check_contract.json').read_text())['expected_tests']
    suite = unittest.defaultTestLoader.discover(str(ROOT/'checks'), pattern='test_*.py', top_level_dir=str(ROOT))
    log = io.StringIO()
    temporary = output / 'temporary'
    temporary.mkdir(exist_ok=False)
    previous = tempfile.tempdir
    try:
        tempfile.tempdir = str(temporary.resolve())
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    finally:
        tempfile.tempdir = previous
    (output/'unittest.log').write_text(log.getvalue(), encoding='utf-8')
    return dict(passed=result.wasSuccessful() and result.testsRun == expected and not result.skipped,
                expected=expected, tests_run=result.testsRun,
                failures=[dict(test=str(t), traceback=error) for t,error in result.failures],
                errors=[dict(test=str(t), traceback=error) for t,error in result.errors],
                skipped=[dict(test=str(t), reason=reason) for t,reason in result.skipped])


def bounded_workflow(output):
    import torch
    torch.set_num_threads(1)
    from chapter3_bser.learning.suite import prepare, run, summarize_suite
    from chapter3_bser.learning.train import load_for_evaluation
    from chapter3_bser.learning.checkpoint import state_digest
    from chapter3_bser.learning.provenance import verify_reviewed_changes
    scenes = json.loads((ROOT/'verification/mechanism_scenes.json').read_text())['scenarios']
    normal = copy.deepcopy(next(s for s in scenes if '_normal_' in s['scenario_id']))
    success = copy.deepcopy(next(s for s in scenes if '_success_' in s['scenario_id']))
    success.update(scenario_role='eval', scenario_split='eval', scenario_id='rebuild_heldout_success', scenario_seed=910827)
    train_manifest=output/'train.json'; eval_manifest=output/'evaluation.json'
    write(train_manifest, dict(scenarios=[normal])); write(eval_manifest, dict(scenarios=[success]))
    configs={}
    for label in ('b2','b3'):
        c=json.loads((ROOT/f'configs/{label}_mechanism.json').read_text())
        c['rl'].update(batch_size=4,warmup_steps=4)
        c.update(scenario_manifest=str(train_manifest), scenario_manifest_sha256=hashlib.sha256(train_manifest.read_bytes()).hexdigest())
        configs[label]=output/f'{label}.json'; write(configs[label], c)
    root=output/'collision_terminal/suite'
    prepare(root, eval_manifest, seeds=[2729], mechanism=True, b2_config=configs['b2'], b3_config=configs['b3'])
    launch=run(root,stage='all')
    summary=summarize_suite(root, output/'suite_report')
    if not summary['complete'] or len(summary['groups'])!=3:
        raise RuntimeError('bounded suite did not close all three methods')
    checkpoints=[]
    for arm in ('D2_B2','D2_B3'):
        path=root/f'jobs/{arm}_seed2729/train/checkpoint_final.pt'
        model,c,payload=load_for_evaluation(path,allow_mechanism=True)
        if model.niter!=4 or payload['optimizer_updates']!=4 or payload['critic_optimizer_steps']!=32:
            raise RuntimeError('trainer/optimizer counts do not match four complete rounds')
        if state_digest(model.training_state_dict())!=payload['model_state_sha256']:
            raise RuntimeError('checkpoint state roundtrip changed')
        checkpoints.append(dict(arm=arm,path=str(path),niter=model.niter,actor_steps=model.actor_updates,roundtrip_exact=True))
    # A full source copy catches late source drift at the actual evaluation end.
    import shutil
    clone=output/'source_negative_copy';clone.mkdir()
    # Copy the complete pinned runtime inventory, never the output directory
    # (which may itself be inside ROOT/work) or any user-retained models.
    from chapter3_bser.learning.provenance import source_files
    inventory=set(source_files(ROOT))
    inventory.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'provenance').glob('*manifest.json'))
    inventory.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'verification').rglob('*') if p.is_file())
    inventory.add('provenance/active_manifest.json')
    for name in sorted(inventory):
        target=clone/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    collision=copy.deepcopy(next(s for s in scenes if '_collision_' in s['scenario_id']))
    collision.update(scenario_role='eval',scenario_split='eval',scenario_id='rebuild_heldout_collision',scenario_seed=910828)
    collision['target_initial_position'][0]-=.5;collision['target_position'][0]-=.5
    collision_manifest=output/'collision_eval.json';write(collision_manifest,dict(scenarios=[collision]))
    code='''import json,sys\nfrom pathlib import Path\nimport torch\ntorch.set_num_threads(1)\nfrom unittest.mock import patch\nfrom chapter3_bser.learning import evaluate as e\nfrom chapter3_bser.learning.runtime import RepairedMissionRuntime\nclass Drift(RepairedMissionRuntime):\n def advance(self,model,**kwargs):\n  row=super().advance(model,**kwargs)\n  with (Path.cwd()/'core/algorithms/noise.py').open('a') as f:f.write('\\n# disposable negative control\\n')\n  return row\noutput=Path(sys.argv[3])\nrejected=False\nwith patch.object(e,'RepairedMissionRuntime',Drift):\n try:e.evaluate(sys.argv[1],sys.argv[2],output,mechanism=True)\n except ValueError as error:rejected='inventory mismatch' in str(error)\nif not rejected or (output/'evaluation_identity.json').exists():raise RuntimeError('late source drift was published as success')\nprint(json.dumps(dict(rejected=True,completed_receipt_absent=True,physical_steps=1)))\n'''
    command=[sys.executable,'-c',code,checkpoints[-1]['path'],str(collision_manifest),str(output/'collision_terminal/source_drift_eval')]
    child=subprocess.run(command,cwd=clone,capture_output=True,env=os.environ.copy())
    (output/'source_negative.stdout').write_bytes(child.stdout);(output/'source_negative.stderr').write_bytes(child.stderr)
    if child.returncode:raise RuntimeError(child.stderr.decode('utf-8',errors='replace'))
    return dict(passed=True,suite=launch,summary=summary,checkpoints=checkpoints,source_negative=json.loads(child.stdout),
                actual_training_episodes=2,training_environment_steps=16,max_complete_updates_per_model=4,
                evaluation_episodes=4,evaluation_environment_steps=19,additional_physical_steps=35,
                fixture_thresholds=dict(batch_size=4,warmup_steps=4),default_configs_unchanged=True,source=verify_reviewed_changes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--rollout',action='store_true')
    parser.add_argument('--map-workflow',action='store_true')
    parser.add_argument('--standby-workflow',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(executed=False,output=args.output,rollout=args.rollout,map_workflow=args.map_workflow,standby_workflow=args.standby_workflow)));return
    if sum((args.rollout,args.map_workflow,args.standby_workflow))>1:parser.error('choose one bounded workflow within the shared episode budget')
    output=Path(args.output).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'work'):
        raise ValueError('in-project verification output must stay under work/')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='auv.active.checks.v1',passed=False,python=sys.version,platform=platform.platform(),command=sys.argv)
    try:
        result['structural']=structural();write(output/'structural.json',result['structural'])
        result['regression']=regression(output);write(output/'regression.json',result['regression'])
        if not result['regression']['passed']:raise RuntimeError('regression failed, skipped, or test inventory incomplete')
        if args.rollout:
            result['workflow']=bounded_workflow(output);write(output/'workflow.json',result['workflow'])
        if args.map_workflow:
            from tools.check_map_training import run as check_maps
            result['workflow']=check_maps(output/'native_maps',bounded=(ROOT/'provenance/map_bounded_manifest.json').is_file())
            write(output/'workflow.json',result['workflow'])
            if not result['workflow']['passed']:raise RuntimeError('native map workflow failed')
        if args.standby_workflow:
            from tools.check_executor_standby import run as check_standby
            result['workflow']=check_standby(output/'executor_standby')
            write(output/'workflow.json',result['workflow'])
            if not result['workflow']['passed']:raise RuntimeError('executor standby workflow failed')
        result['passed']=True
    except Exception:
        import traceback
        result['traceback']=traceback.format_exc()
    finally:
        write(output/'checks.json',result)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
