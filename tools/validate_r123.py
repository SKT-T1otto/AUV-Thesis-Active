"""Source-matched contracts and four explicit 32-step native mechanism checks."""
import argparse
import ast
import copy
import json
import platform
import traceback
import unittest
from unittest.mock import patch
from project_paths import ROOT
from chapter3_bser.r123.source import identity,write,sha,new_output


def validate(output,*,require_linux=False,native_smoke=False):
    out=new_output(output);out.mkdir(parents=True)
    result=dict(schema='auv.r123.validation.v1',source=identity(),platform=platform.system(),passed=False,
        environment_initializations=0,physical_steps=0,native_smoke_passed=False,
        max_complete_updates_per_temporary_model=4,automatic_long_training=False,
        checkpoint_restore=False,performance_accepted=False)
    try:
        if require_linux and platform.system()!='Linux':raise ValueError('actual Linux required')
        forbidden={'ground_truth_obstacles','_raycast_ground_truth','_nearest_obstacle_distance',
            '_flow_at','_flow_phase_x','_flow_phase_y','_obstacle_lower','_obstacle_upper'}
        for p in (ROOT/'chapter3_bser/r123').glob('*.py'):
            for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
                if isinstance(n,ast.Attribute) and n.attr in forbidden:raise ValueError('hidden-world access: '+str(p))
                if isinstance(n,ast.ImportFrom) and (n.module or '').startswith('tools'):raise ValueError('runtime imports tools')
        from checks import test_r123,test_path_option,test_path_option_compute,test_path_option_scenarios
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m)
            for m in (test_r123,test_path_option,test_path_option_compute,test_path_option_scenarios))
        with (out/'synthetic.log').open('x',encoding='utf-8') as f:
            with patch('core.env.mission_env.MissionCoreEnv.__init__',side_effect=AssertionError('native env forbidden in synthetic tests')):
                tests=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        result['synthetic']=dict(tests=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped))
        if not tests.wasSuccessful() or tests.skipped:raise ValueError('synthetic contracts failed')
        import torch
        from chapter3_bser.r123.learner import Learner
        from chapter3_bser.r123.options import ARMS,options
        torch.set_num_threads(1)
        sizes=[]
        reference=None
        for arm in ARMS:
            model=Learner(arm=arm)
            weights={k:sha_tensor(v) for k,v in model.policy.state_dict().items()}
            if reference is not None and weights!=reference:raise ValueError('arm initializations differ')
            reference=weights
            sizes.append(dict(arm=arm,policy=sum(p.numel() for p in model.policy.parameters()),
                twin_critics=sum(p.numel() for p in model.critics.parameters())))
        write(out/'network_sizes.json',sizes)
        if native_smoke:
            from tools.path_option_experiment import generate
            from chapter3_bser.r123.workflow import one_episode,task_config
            config=task_config(json.loads((ROOT/'configs/b2.json').read_text(encoding='utf-8')))
            config['max_steps']=32
            scene=generate(1,960001,'eval')['scenarios'][0]
            for i,arm in enumerate(ARMS):
                print(f'[r123 validation] native {i+1}/4: {arm}, at most32 steps, zero updates',flush=True)
                case=copy.deepcopy(scene);case['max_steps']=32
                write(out/f'initialization_{i:02d}.json',dict(arm=arm,step_cap=32,updates=0))
                result['environment_initializations']+=1
                def count_step(step):result['physical_steps']+=1
                model=Learner(options(),arm)
                raw,m,complete=one_episode(config,case,options(),model,arm=arm,seed=960001,episode_id=i,step_hook=count_step)
                if m['steps']>32 or model.updates:raise ValueError('native budget exceeded')
                write(out/f'episode_{i:02d}.json',raw);write(out/f'metric_{i:02d}.json',m)
            result['native_smoke_passed']=True
        if identity()!=result['source']:raise ValueError('source changed during validation')
        result['passed']=True
    except Exception:result['traceback']=traceback.format_exc()
    result['artifacts']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'validation.json',result)
    return result


def sha_tensor(t):
    from chapter3_bser.r123.workflow import sha_tensor as f
    return f(t)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--require-linux',action='store_true')
    p.add_argument('--native-smoke',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    r=validate(a.output,require_linux=a.require_linux,native_smoke=a.native_smoke) if a.execute else dict(executed=False,requested=vars(a))
    print(json.dumps(r,indent=2,allow_nan=False))
    if a.execute and not r['passed']:raise SystemExit(1)
    return r


if __name__=='__main__':main()
