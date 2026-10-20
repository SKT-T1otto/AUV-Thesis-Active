"""Bounded interface parity against pre-change Active data; never restore a model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
from project_paths import ROOT


def run(output, baseline, *, capture):
    import numpy as np
    import torch
    from chapter3_bser.learning.runtime import RepairedMissionRuntime
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.learning.provenance import verify_reviewed_changes
    from chapter3_bser.experiments.coupled_search_v1.config import DEFAULT
    identity = verify_reviewed_changes()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):
        raise ValueError('map check output must be under Active/work')
    golden = json.loads(Path(baseline).read_text(encoding='utf-8'))
    if [row['scene_index'] for row in golden] != [0, 3]:
        raise ValueError('expected the two pre-change interface scenes')
    output.mkdir(parents=True, exist_ok=False)
    scenes = json.loads((ROOT/'verification/mechanism_scenes.json').read_text())['scenarios']
    torch.set_num_threads(1)
    results = []
    for expected in golden:
        index = expected['scene_index']
        config = json.loads((ROOT/'configs/b2_mechanism.json').read_text()); config['max_steps'] = 4
        scene = copy.deepcopy(scenes[index]); scene['max_steps'] = 4
        runtime = RepairedMissionRuntime(config, scene, seed=82729, episode_id=index,
                    map_research=dict(DEFAULT, enabled=True) if capture else None)
        try:
            trace = []

            def snapshot(row=None):
                env = runtime.env.unwrapped
                names = ('_agent_pos', '_agent_vel', '_agent_acc', '_nav_targets', '_task_target', '_prev_acc', '_agent_task_known')
                state = {n: getattr(env,n).detach().cpu().numpy().tolist() for n in names}
                state.update(step=runtime.step, terminal=runtime.terminal, found=bool(env.task_found),
                             occupancy=env.map_module.occupancy_logodds.cpu().numpy().tolist(),
                             belief=env.map_module.belief_map.cpu().numpy().tolist(),
                             observations=np.asarray(runtime.observations).tolist(), guidance=repr(runtime.guidance))
                state['rng'] = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().tolist()))).hexdigest()
                if row:
                    state['transition'] = {k: np.asarray(row[k]).tolist() for k in ('actions','rewards','dones','action_enabled','next_action_enabled')}
                trace.append(state)

            snapshot()
            while not runtime.terminal:
                if runtime.step >= 4:
                    raise RuntimeError('map check exceeded its four-step episode bound')
                snapshot(runtime.advance(ZeroResidualPolicy(), explore=False))
            if trace != expected['trace']:
                differences = [(i, key) for i,(a,b) in enumerate(zip(trace, expected['trace']))
                               for key in a.keys() | b.keys() if a.get(key) != b.get(key)]
                raise RuntimeError('interface parity mismatch: '+repr(differences[:20]))
            session = runtime.map_research_session
            if capture:
                summary = session.summary()
                if not summary['valid'] or summary['counts']['physical_transitions'] != runtime.step:
                    raise RuntimeError('invalid/incomplete public evidence: '+repr(summary))
                if summary['counts'].get('negative_observations', 0) != (0 if index == 3 else runtime.step):
                    raise RuntimeError('negative evidence does not follow detector completion')
                (output/f'audit_scene{index}.json').write_text(json.dumps(list(session.records),indent=2),encoding='utf-8')
            results.append(dict(scene_index=index, steps=runtime.step, exact_parity=True,
                                public_evidence=None if session is None else session.summary()))
        finally:
            runtime.close()
    if verify_reviewed_changes() != identity:
        raise ValueError('source changed during map interface verification')
    result = dict(passed=True, source_identity=identity, capture=capture, episodes=2,
                  physical_steps=sum(r['steps'] for r in results), results=results,
                  baseline_sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
                  model_restored=False, optimizer_updates=0, performance_passed=False)
    (output/'parity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        result = dict(executed=False, capture=args.capture, output=args.output, episodes=2, max_steps=4)
    else:
        result = run(args.output, args.baseline, capture=args.capture)
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    main()
