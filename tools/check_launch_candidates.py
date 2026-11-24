"""Two-step development replay of native, single-offer and two-offer departures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from chapter3_bser.launch_assignment import LaunchAssignmentMission
from chapter3_bser.local_ctde.policy import GoalPolicy
from chapter3_bser.neighbor_ctde.mission import NeighborMission
from chapter3_bser.neighbor_ctde.observations import CONTEXT_DIM
from chapter3_bser.neighbor_ctde.policy import NeighborPolicy
from project_paths import ROOT
from tools.launch_assignment_study import CONFIG, file_hash, read, scene_identity, stream_seed


LOCK = ROOT / 'provenance/launch_candidates_v1.json'
MANIFEST = ROOT / 'verification/launch_assignment_v1/manifest.json'
SCENE = 'r123_expand_20261119_0003'


class LastOption:
    """Exercise submission of the alternate. This is a mechanism-test selector."""
    def __call__(self, menu):
        return len(menu.points) - 1


def generators(streams, family, kind, repeat=0):
    return [torch.Generator().manual_seed(stream_seed(streams, family, kind, i, repeat))
            for i in range(3)]


def parameter_hash(actor):
    result = hashlib.sha256()
    for name, param in sorted(actor.state_dict().items()):
        result.update(name.encode())
        result.update(param.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def make_actor(width, reference, use_checkpoint):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reference['model_seed'])
        actor = NeighborPolicy(GoalPolicy(width))
    if use_checkpoint:
        path = ROOT / reference['path']
        if file_hash(path) != reference['file_sha256']:
            raise ValueError('frozen proposer file hash mismatch')
        data = torch.load(path, map_location='cpu', weights_only=True)
        if (data['schema'], data['local_dim'], data['context_dim'], data['width']) != (
                'ch3.neighbor_search_goal_actor.v1', 28, CONTEXT_DIM, width):
            raise ValueError('frozen proposer architecture mismatch')
        actor.load_state_dict(data['actor'], strict=True)
        if parameter_hash(actor) != reference['actor_parameters_sha256']:
            raise ValueError('frozen proposer parameter hash mismatch')
    return actor.eval().requires_grad_(False)


def verify_sources():
    lock = read(LOCK)
    successor = None
    for name, expected in lock['source_lf_sha256'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            from tools.launch_intervention_source import reviewed_successor
            successor = reviewed_successor(name, expected, normalized=True)
            if not successor:
                raise ValueError('candidate source mismatch: ' + name)
    return successor or file_hash(LOCK)


def replay(use_checkpoint=False):
    source_hash = verify_sources()
    study = read(CONFIG)
    native = study['native_inputs']
    config, options = [read(ROOT / native[k]) for k in ('config', 'options')]
    scenes = read((ROOT / native['config']).with_name('scenes.json'))['scenarios']
    scene = next(s for s in scenes if s['scenario_id'] == SCENE)
    physical = scene_identity(scene)['physical']
    families = [f for f in read(MANIFEST)['families']
                if any(m['physical'] == physical for m in f['members'])]
    if len(families) != 1 or families[0]['split'] != 'development':
        raise ValueError('mechanism replay requires the registered development family')
    family = families[0]['family_id']
    seeds = study['random_streams']
    torch.set_num_threads(1)
    actor = make_actor(config['rl']['hidden_dim'], study['candidates']['proposer_reference'], use_checkpoint)
    actor_hash = parameter_hash(actor)
    radius = study['problem']['information_boundary']['link_radius_m']
    arms = {}
    for name in ('native', 'single', 'two_first', 'two_last'):
        originals = generators(seeds, family, 'original_offer')
        continuation = generators(seeds, family, 'continuation')
        if name == 'native':
            mission = NeighborMission(config, options, scene, radius=radius)
            # Two steps are shorter than the native ten-step re-request interval.
            rollout_streams = originals
        else:
            extra = None if name == 'single' else generators(seeds, family, 'extra_offer')
            mission = LaunchAssignmentMission(config, options, scene, radius=radius, originals=originals,
                                              extras=extra, selector_factory=LastOption if name == 'two_last' else None)
            rollout_streams = continuation
        try:
            episode = mission.rollout(actor, rollout_streams, cap=2)
        finally:
            mission.close()
        if any(d['step'] != 0 for d in episode['decisions']):
            raise RuntimeError('unexpected non-departure draw in the two-step fixture')
        arms[name] = {k: episode[k] for k in ('steps', 'frames', 'rewards', 'decisions',
                                             'projection_queries', 'execution_route_requests')}
    native_arm, single, first, last = arms.values()
    if single['frames'] != native_arm['frames'] or single['rewards'] != native_arm['rewards']:
        raise RuntimeError('single-offer replay disagrees with the native commands or rewards')
    single_draws = single['decisions']
    if not all([d['agent'] for d in arm['decisions']] == [0, 1, 2] for arm in arms.values()):
        raise RuntimeError('expected one departure decision per search boat')
    for old, one, two, other in zip(native_arm['decisions'], single_draws, first['decisions'], last['decisions']):
        draw = one['departure']['draws'][0]
        if (tuple(old['latent']), tuple(old['raw_goal']), old['proposed'], old['accepted']) != (
                draw['latent'], draw['raw_goal'], one['proposed'], one['accepted']):
            raise RuntimeError('reference draw or native admission changed')
        if two['departure']['draws'][0] != draw or two['departure']['draws'] != other['departure']['draws']:
            raise RuntimeError('selector or extra stream changed the predecision offers')
    same_firsts = all(a['proposed'] == b['proposed'] for a, b in zip(single_draws, first['decisions']))
    if same_firsts and (first['frames'] != single['frames'] or first['rewards'] != single['rewards']):
        raise RuntimeError('unused alternatives changed the native prefix')
    for arm in (single, first, last):
        accounted = sum(d['departure']['preview_queries'] for d in arm['decisions'])
        if arm['projection_queries'] != accounted:
            raise RuntimeError('unaccounted projection queries')
    if native_arm['projection_queries'] != single['projection_queries'] or parameter_hash(actor) != actor_hash:
        raise RuntimeError('single-offer resource use or model parameters changed')
    return dict(schema='auv.launch_candidates.replay.v1', source_lock_sha256=source_hash,
                scene=SCENE, family=family, split='development', steps_per_arm_cap=2,
                policy='frozen_checkpoint' if use_checkpoint else 'initialized_fixture',
                actor_parameters_sha256=actor_hash, checkpoint_loads=int(use_checkpoint), training_updates=0,
                same_first_assignments=same_firsts, physical_steps=sum(a['steps'] for a in arms.values()), arms=arms)


def write_receipt(path, result):
    data = (json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        staged = Path(scratch) / 'receipt.json'
        staged.write_bytes(data)
        if os.name == 'nt':
            os.rename(staged, path)  # atomic no-replace on Windows
        else:
            os.link(staged, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--frozen-actor', action='store_true', help='explicitly load the hash-checked study proposer')
    parser.add_argument('--output', type=Path, help='new JSON receipt file')
    args = parser.parse_args()
    if not args.execute:
        print('Development 0003 only; 4 arms x 2 steps; initialized actor by default; zero training. '
              'Use --execute --output FILE to run.')
        return
    if args.output is None or args.output.exists():
        parser.error('--execute requires a new --output file')
    result = replay(args.frozen_actor)
    write_receipt(args.output, result)
    print(f"{result['physical_steps']} physical steps, 0 training updates. {args.output}")


if __name__ == '__main__':
    main()
