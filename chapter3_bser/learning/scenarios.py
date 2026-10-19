"""Explicit manifest export; preserve generated physical tasks and separate roles."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from chapter3_bser.learning.checkpoint import digest, write_json
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review

ROLE_TO_GENERATOR = {'train': 'train', 'eval': 'validation', 'smoke_train': 'smoke_train', 'smoke_eval': 'smoke_validation'}


def validate_manifest(manifest, config, *, evaluation=False, mechanism=False):
    scenes = manifest.get('scenarios')
    if not isinstance(scenes, list) or not scenes:
        raise ValueError('a nonempty scenario list is required')
    roles = ('eval', 'smoke_eval') if evaluation else ('train', 'smoke_train')
    ids, seeds, fingerprints = set(), set(), set()
    for scene in scenes:
        name, seed = scene.get('scenario_id'), scene.get('scenario_seed')
        if (not isinstance(name, str) or not name.strip() or name in ids or type(seed) is not int or not 0 <= seed < 2**63 or seed in seeds
                or scene.get('scenario_role') not in roles or scene.get('scenario_split') != scene['scenario_role']
                or scene.get('scenario_profile') != config['profile'] or scene.get('max_steps') != config['max_steps']
                or scene.get('protocol') not in ('CH3_UNKNOWN_MAP_V1', 'ch3_unknown_map_v1')):
            raise ValueError('scenario ID/seed/role/profile/horizon/protocol contract mismatch')
        fingerprint = scenario_fingerprint(scene)
        if fingerprint in fingerprints:
            raise ValueError('repeated physical scenario under different metadata')
        ids.add(name); seeds.add(seed); fingerprints.add(fingerprint)
    if manifest.get('scenario_count', len(scenes)) != len(scenes):
        raise ValueError('scenario count mismatch')
    if mechanism and (len(scenes) > 8 or config['max_steps'] > 32 or len(scenes) * config['max_steps'] > 256):
        raise ValueError('mechanism evaluation requires <=8 scenes, <=32 steps each and <=256 total steps')
    return scenes


def export(output, *, split='eval', count=100, seed=92729, max_steps=400):
    identity = verify_reviewed_changes()
    if split not in ROLE_TO_GENERATOR or type(count) is not int or count < 1 or type(seed) is not int or not 0 <= seed < 2**31 or type(max_steps) is not int or max_steps < 1:
        raise ValueError('invalid scenario export parameters')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    origin = ROLE_TO_GENERATOR[split]
    generated = build_scenario_manifests(count=count, generator_seed=seed, split=origin, profiles=['M20_MOVING_UNKNOWN_MULTI'])['M20_MOVING_UNKNOWN_MULTI']
    result = copy.deepcopy(generated)
    for scene in result['scenarios']:
        scene.update(scenario_role=split, scenario_split=split, max_steps=max_steps)
    result.update(scenario_role=split, scenario_split=split, max_steps=max_steps,
                  export_schema='auv.active.manifest.v1', source_review=identity,
                  generation_origin=dict(split=origin, seed=seed, manifest_sha256=digest(generated)), role_mapping_explicit=True)
    validate_manifest(result, dict(profile='M20_MOVING_UNKNOWN_MULTI', max_steps=max_steps), evaluation='eval' in split)
    require_review(identity)
    write_json(output, result)
    return dict(output=str(output), scenarios=count, split=split, seed=seed, max_steps=max_steps,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(), executed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=ROLE_TO_GENERATOR, default='eval')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=92729)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--execute', action='store_true')
    args = vars(parser.parse_args(argv)); execute = args.pop('execute')
    result = export(**args) if execute else dict(executed=False, **args)
    print(json.dumps(result, indent=2)); return result
