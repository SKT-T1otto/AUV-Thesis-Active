"""Independent source + synthetic contracts; native smoke is explicit and bounded."""
import argparse
import ast
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import unittest
from project_paths import ROOT
from chapter3_bser.path_option.source import identity, write, sha, new_output


def validate(output, *, require_linux=False, native_smoke=False):
    out = new_output(output); out.mkdir(parents=True)
    result = dict(schema='auv.path_option.validation.v1', passed=False,
        source=identity(), platform=platform.system(), environment_initializations=0,
        physical_steps=0, native_smoke_passed=False, max_complete_updates_per_temporary_model=4,
        automatic_long_training=False, checkpoint_restore=False, performance_accepted=False)
    try:
        if require_linux and platform.system() != 'Linux':
            raise ValueError('--require-linux requires an actual Linux host')
        if native_smoke and platform.system() != 'Linux':
            raise ValueError('native smoke reserved for explicit manual Linux execution')
        forbidden = {'ground_truth_obstacles', '_raycast_ground_truth', '_nearest_obstacle_distance',
            '_flow_at', '_flow_phase_x', '_flow_phase_y', '_obstacle_lower', '_obstacle_upper'}
        for p in (ROOT/'chapter3_bser/path_option').glob('*.py'):
            tree = ast.parse(p.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and n.attr in forbidden:
                    raise ValueError('hidden-world attribute in new runtime: '+str(p))
                if isinstance(n, ast.ImportFrom) and (n.module or '').startswith('tools'):
                    raise ValueError('runtime imports tools')
        from checks import test_path_option, test_path_option_compute
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module)
                                   for module in (test_path_option, test_path_option_compute))
        # Fail closed if synthetic tests accidentally try to construct native environments.
        with (out/'synthetic.log').open('x', encoding='utf-8') as f:
            with patch('core.env.mission_env.MissionCoreEnv.__init__', side_effect=AssertionError('native environment forbidden in synthetic suite')):
                tests = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
        result['synthetic'] = dict(tests=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors), skipped=len(tests.skipped))
        if not tests.wasSuccessful() or tests.skipped:
            raise ValueError('synthetic contracts failed; see synthetic.log')
        import torch
        from chapter3_bser.path_option.learner import Learner
        from chapter3_bser.path_option.networks import tensor_batch
        sizes = []
        for variant in ('A', 'B', 'C', 'D', 'fixed_query'):
            model = Learner(variant=variant)
            b = tensor_batch([test_path_option.fixture()])
            with torch.no_grad():
                model.distributions(b, [test_path_option.fixture()])
                started = time.perf_counter()
                for _ in range(5):
                    model.distributions(b, [test_path_option.fixture()])
            sizes.append(dict(variant=variant,
                policy_parameters=sum(p.numel() for p in model.policy.parameters()),
                twin_critic_parameters=sum(p.numel() for p in model.critics.parameters()),
                synthetic_decision_seconds=(time.perf_counter()-started)/5,
                scope='synthetic single-agent 8-goal query; not end-to-end environment throughput'))
        write(out/'network_sizes.json', sizes)
        if native_smoke:
            import copy
            import torch
            from tools.path_option_experiment import generate
            from chapter3_bser.path_option.workflow import one_episode, task_config
            from chapter3_bser.path_option.options import DEFAULTS
            from chapter3_bser.path_option.learner import Learner
            torch.set_num_threads(1)
            c = task_config(json.loads((ROOT/'configs/b2.json').read_text(encoding='utf-8')))
            c['max_steps'] = 32
            scenes = generate(2, 960001, 'eval')['scenarios']
            for i, scene in enumerate(scenes):
                scene = copy.deepcopy(scene); scene['max_steps'] = 32
                write(out/f'initialization_{i+1:02d}.json', dict(case='rule' if i == 0 else 'A_untrained', limit=32))
                result['environment_initializations'] += 1
                def count_step(step):
                    result['physical_steps'] += 1
                raw, metric, _ = one_episode(c, scene, DEFAULTS, None if i == 0 else Learner(),
                    seed=960001+i, episode_id=i, step_hook=count_step)
                if metric['steps'] > 32:
                    raise ValueError('native physical step cap exceeded')
                write(out/f'episode_{i+1:02d}.json', raw)
            result['native_smoke_passed'] = True
        if identity() != result['source']:
            raise ValueError('source changed during checks')
        result['passed'] = True
    except Exception:
        result['traceback'] = traceback.format_exc()
    result['artifacts'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'validation.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--require-linux', action='store_true')
    p.add_argument('--native-smoke', action='store_true'); p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    r = validate(a.output, require_linux=a.require_linux, native_smoke=a.native_smoke) if a.execute else dict(executed=False, requested=vars(a))
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if a.execute and not r['passed']:
        raise SystemExit(1)
    return r


if __name__ == '__main__':
    main()
