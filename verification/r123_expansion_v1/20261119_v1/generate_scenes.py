"""Freeze 24 outcome-blind new M20 scenes using the unchanged generator."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.scenarios import validate_manifest
from chapter3_bser.r123_task_sync.source import identity

SEEDS = (2026111901, 2026112901, 2026113901, 2026114901)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source = identity()
    old = ROOT / '3090结果/11.16/r123_20261116_103339_2030716/screen'
    baseline = ROOT / '3090结果/11.18/r123_tracking_20261117_121047_2475513'
    config = read(baseline / 'plan/config.json')
    options = read(baseline / 'plan/options.json')
    forbidden = set()
    origins = []
    for name in ('training.json', 'development.json', 'test.json'):
        p = old / name
        scenes = read(p)['scenarios']
        forbidden.update(scenario_fingerprint(s) for s in scenes)
        origins.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), count=len(scenes),
                            use='fingerprint exclusion only; no outcome or policy selection'))
    scenes = []
    generation = []
    for batch, seed in enumerate(SEEDS):
        print(f'[new scenes] batch={batch + 1}/4 seed={seed} count=6; no mission rollout', flush=True)
        generated = build_scenario_manifests(count=6, generator_seed=seed, split='validation',
            profiles=['M20_MOVING_UNKNOWN_MULTI'])['M20_MOVING_UNKNOWN_MULTI']
        original = copy.deepcopy(generated)
        for j, scene in enumerate(generated['scenarios']):
            n = len(scenes)
            scene.update(scenario_id=f'r123_expand_20261119_{n:04d}', scenario_role='eval', scenario_split='eval')
            fingerprint = scenario_fingerprint(scene)
            if fingerprint in forbidden:
                raise ValueError('physical overlap: do not silently resample')
            forbidden.add(fingerprint)
            scenes.append(scene)
        canonical = json.dumps(original, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        generation.append(dict(batch=batch, seed=seed, count=6, original_manifest_sha256=hashlib.sha256(canonical).hexdigest()))
    manifest = dict(schema='auv.r123.expanded_development_scenes.v1', scenario_profile='M20_MOVING_UNKNOWN_MULTI',
        scenario_role='eval', scenario_split='eval', scenario_count=24, max_steps=400,
        purpose='expanded development screening, not final held-out test', generation=generation,
        selection='all generated scenes; no performance filtering or seed retry', scenarios=scenes)
    validate_manifest(manifest, config, evaluation=True)
    target = ROOT / 'configs/r123_expansion_v1'
    target.mkdir(parents=True, exist_ok=False)
    write(target / 'config.json', config)
    write(target / 'options.json', options)
    write(target / 'scenes.json', manifest)
    old_fingerprints = sorted({scenario_fingerprint(s) for name in ('training.json', 'development.json', 'test.json')
                              for s in read(old / name)['scenarios']})
    write(target / 'exclusions.json', dict(origins=origins, fingerprints=old_fingerprints))
    write(OUT / 'generation.json', dict(completed=True, source=source, new_scenes=24, regression_scenes=6,
        generation=generation, environment_initializations=0, physical_steps=0, training_updates=0,
        inputs={p.name: sha(p) for p in target.iterdir() if p.is_file()}, exclusions=len(old_fingerprints)))
    print(json.dumps(dict(generated=True, count=24, environment_initializations=0, physical_steps=0)))


if __name__ == '__main__':
    main()
