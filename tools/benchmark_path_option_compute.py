"""Bounded synthetic equivalence and CPU timing; no simulator or weight restore."""
import argparse
import io
import json
import platform
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from chapter3_bser.path_option.source import identity, new_output, sha, write
from checks import test_path_option_compute as check
from chapter3_bser.path_option.networks import tensor_batch


def benchmark(repeats):
    cases = []
    for length in (16, 64):
        for variant in ('A', 'B'):
            measurements = []
            for repeat in range(repeats):
                models = check.pair(variant)
                rr = check.rows(length=length)
                for model in models:
                    check.fill(model, rr)
                inference = [[], []]; updates = [[], []]
                batch = tensor_batch(rr[:1])
                with torch.no_grad():
                    for model in models:
                        for _ in range(5):
                            model.distributions(batch, rr[:1])
                    for i in range(20):
                        for side in ((0, 1) if (i+repeat) % 2 == 0 else (1, 0)):
                            start = time.perf_counter()
                            models[side].distributions(batch, rr[:1])
                            inference[side].append(time.perf_counter()-start)
                # Exactly four complete updates per temporary learner, no warmup
                # updates or checkpoint restore. Pair order alternates per step.
                for i in range(4):
                    losses = [None, None]
                    for side in ((0, 1) if (i+repeat) % 2 == 0 else (1, 0)):
                        start = time.perf_counter(); losses[side] = models[side].update()
                        updates[side].append(time.perf_counter()-start)
                    check.close(losses[0], losses[1], 'benchmark_losses')
                check.close(check.model_state(models[0]), check.model_state(models[1]), 'benchmark_state')
                measurements.append(dict(inference_seconds=inference, update_seconds=updates,
                                         complete_updates_per_model=4))
            median = lambda key, side: float(np.median([x for m in measurements for x in m[key][side]]))
            before_i, after_i = [median('inference_seconds', i) for i in (0, 1)]
            before_u, after_u = [median('update_seconds', i) for i in (0, 1)]
            case = dict(variant=variant, vertices_per_path=length, batch_size=32, repeats=repeats,
                inference_before_ms=before_i*1000, inference_after_ms=after_i*1000,
                inference_speedup=before_i/after_i, update_before_ms=before_u*1000,
                update_after_ms=after_u*1000, update_speedup=before_u/after_u,
                synthetic_network_step_before=4*before_i+before_u/4,
                synthetic_network_step_after=4*after_i+after_u/4, raw=measurements)
            cases.append(case)
            print(json.dumps({k: v for k, v in case.items() if k != 'raw'}), flush=True)
    return cases


def run(output, repeats):
    source = identity(); out = new_output(output); out.mkdir(parents=True)
    torch.set_num_threads(1)
    result = dict(source=source, platform=platform.platform(), torch=torch.__version__,
        cpu=platform.processor(), threads=1, environment_initializations=0, physical_steps=0,
        research_training=False, checkpoint_restore=False, max_updates_per_temporary_model=4,
        scope='Synthetic CPU network only; not native throughput or research performance',
        tolerances=dict(rtol=1e-5, atol=1e-7), passed=False)
    # Enforce the no-environment contract throughout both checks and timings.
    with patch('core.env.mission_env.MissionCoreEnv.__init__', side_effect=AssertionError('native environment forbidden')):
        stream = io.StringIO()
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(check))
        (out/'equivalence.log').write_text(stream.getvalue(), encoding='utf-8')
        result['equivalence'] = dict(tests=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors))
        if tests.wasSuccessful():
            result['cases'] = benchmark(repeats)
            result['passed'] = True
    result['maximum_absolute_errors'] = check.ERRORS
    result['source_unchanged'] = identity() == source
    result['passed'] &= result['source_unchanged']
    result['artifacts'] = {'equivalence.log': sha(out/'equivalence.log')}
    write(out/'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--execute', action='store_true')
    p.add_argument('--repeats', type=int, choices=(1, 2, 3), default=3)
    a = p.parse_args()
    if a.execute:
        run(a.output, a.repeats)
    else:
        print(json.dumps(dict(executed=False, requested=vars(a))))


if __name__ == '__main__':
    main()
