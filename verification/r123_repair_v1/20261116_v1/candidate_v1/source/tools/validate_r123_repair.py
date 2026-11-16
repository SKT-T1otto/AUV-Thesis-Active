"""Rule-repair validation: synthetic regressions, optional five32-step cases."""
import argparse
import ast
import copy
from contextlib import redirect_stdout,redirect_stderr
import json
import platform
import traceback
import unittest
from unittest.mock import patch
from project_paths import ROOT
from chapter3_bser.r123_repair.source import identity,write,sha,new_output


def validate(output,*,require_linux=False,native_smoke=False):
    out=new_output(output);out.mkdir(parents=True)
    result=dict(schema='auv.r123.rule_repair.validation.v1',source=identity(),platform=platform.system(),passed=False,
        native_smoke_passed=False,environment_initializations=0,physical_steps=0,native_training_updates=0,
        checkpoint_restore=False,automatic_long_training=False,performance_accepted=False,
        max_complete_updates_per_temporary_model=4)
    try:
        if require_linux and platform.system()!='Linux':raise ValueError('actual Linux host required')
        forbidden={'ground_truth_obstacles','_raycast_ground_truth','_nearest_obstacle_distance','_flow_at',
                   '_flow_phase_x','_flow_phase_y','_obstacle_lower','_obstacle_upper'}
        for path in (ROOT/'chapter3_bser/r123_repair').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.Attribute) and node.attr in forbidden:raise ValueError('hidden-world access: '+str(path))
                if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('tools'):raise ValueError('runtime imports tools')
        from checks import test_r123_repair,test_r123,test_path_option,test_path_option_compute,test_path_option_scenarios
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in
            (test_r123_repair,test_r123,test_path_option,test_path_option_compute,test_path_option_scenarios))
        with (out/'synthetic.log').open('x',encoding='utf-8') as f:
            with patch('core.env.mission_env.MissionCoreEnv.__init__',side_effect=AssertionError('native environment forbidden in synthetic suite')),redirect_stdout(f),redirect_stderr(f):
                tests=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        result['synthetic']=dict(tests=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped))
        if not tests.wasSuccessful() or tests.skipped:raise ValueError('synthetic regressions failed')
        if native_smoke:
            import torch
            from tools.path_option_experiment import generate,read
            from chapter3_bser.path_option.workflow import task_config
            from chapter3_bser.r123.options import options
            from chapter3_bser.r123_repair.workflow import one_episode
            torch.set_num_threads(1)
            config=task_config(read(ROOT/'configs/b2.json'));config['max_steps']=32
            scene=generate(1,960001,'eval')['scenarios'][0]
            cases=[('repaired',True),('no_recovery',True),('legacy',False),('repaired',False),('no_recovery',False)]
            for n,(variant,release_probe) in enumerate(cases):
                print(f'[repair validation] case={n+1}/5 variant={variant} release_probe={release_probe} cap=32 updates=0',flush=True)
                record=dict(variant=variant,release_probe=release_probe,max_steps=32,updates=0)
                write(out/f'initialization_{n:02d}.json',record);result['environment_initializations']+=1
                s=copy.deepcopy(scene);s['max_steps']=32
                def count_step(step):result['physical_steps']+=1
                raw,m=one_episode(config,s,options(),variant,seed=960001,episode_id=n,step_hook=count_step,release_probe=release_probe)
                if m['steps']>32:raise ValueError('native budget exceeded')
                if variant!='legacy':
                    if m['frozen_recovery_enters']:raise ValueError('frozen recovery bookkeeping')
                    if variant=='no_recovery' and m['active_recovery_enters']:raise ValueError('deletion control entered recovery')
                write(out/f'episode_{n:02d}.json',raw);write(out/f'metric_{n:02d}.json',m)
            result['native_smoke_passed']=True
        if identity()!=result['source']:raise ValueError('source changed during checks')
        result['passed']=True
    except Exception:result['traceback']=traceback.format_exc()
    result['artifacts']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'validation.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--require-linux',action='store_true')
    p.add_argument('--native-smoke',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    result=validate(a.output,require_linux=a.require_linux,native_smoke=a.native_smoke) if a.execute else dict(executed=False,requested=vars(a))
    print(json.dumps(result,indent=2,allow_nan=False))
    if a.execute and not result['passed']:raise SystemExit(1)
    return result


if __name__=='__main__':main()
