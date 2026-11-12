"""Freeze a preparation-only source evolution, retaining all earlier evidence."""
import argparse
import json
from project_paths import ROOT
from chapter3_bser.path_option.source import sha, digest, legacy_digest, write

CHANGED = {
    'tools/path_option_experiment.py': 'Correct special trajectory direction at unchanged speed; version v2 cases; report preparation progress',
    'tools/validate_path_option.py': 'Add offline scene feasibility regression before native smoke',
    'chapter3_bser/path_option/source.py': 'Bind source v5 without reusing prior validation identities',
}
ADDED = ['checks/test_path_option_scenarios.py', 'tools/build_path_option_scene_manifest.py',
         'docs/PATH_OPTION_SCENE_FIX_20261112.md',
         'verification/path_option_scene_fix_v1/20261112_v1/before.json',
         *['verification/path_option_scene_fix_v1/20261112_v1/before/'+name for name in CHANGED]]


def build():
    target = ROOT/'provenance/path_option_manifest_v5.json'
    evolution = ROOT/'provenance/path_option_scene_evolution_v1.json'
    if target.exists() or evolution.exists():
        raise FileExistsError('scene evolution already frozen')
    parent_name = 'provenance/path_option_manifest_v4.json'
    parent = json.loads((ROOT/parent_name).read_text(encoding='utf-8'))
    if (parent['sha256'] != 'e1477269387a1926d5de250e928cd7914db7affd02fc929edb8416005d5866ef'
            or digest({k: v for k, v in parent.items() if k != 'sha256'}) != parent['sha256']):
        raise ValueError('frozen compute parent changed')
    files = dict(parent['files'])
    for name, expected in files.items():
        actual = legacy_digest(ROOT/name) if name in parent['normalized_existing_dispatch'] else sha(ROOT/name)
        if name not in CHANGED and actual != expected:
            raise ValueError('unreviewed parent change: '+name)
    for name in (*CHANGED, *ADDED, parent_name):
        files[name] = sha(ROOT/name)
    changes = [dict(path=name, before_sha256=parent['files'].get(name), after_sha256=files[name],
        reason=CHANGED.get(name, 'Independent offline regression, frozen parent or repair documentation'))
        for name in sorted(files) if files[name] != parent['files'].get(name)]
    write(evolution, dict(schema='auv.path_option.scene_evolution.v1', parent_source=parent['sha256'],
        changes=changes, special_evaluation_revision='path_option_challenge_v2',
        runtime_algorithm_reward_training_configuration_changed=False,
        feasibility_gate_preserved=True, special_trajectory_direction_changed=True))
    files[evolution.relative_to(ROOT).as_posix()] = sha(evolution)
    data = dict(schema=parent['schema'], parent_commit='ce737240fdb5dd81a1e3ab0950c38da3f0897bb5',
        normalized_existing_dispatch=parent['normalized_existing_dispatch'], files=files,
        parent_source_sha256=parent['sha256'], special_evaluation_revision='path_option_challenge_v2',
        runtime_algorithm_configuration_changed=False, automatic_native_budget_remaining=0,
        old_evidence_relabeling_allowed=False)
    data['sha256'] = digest(data); write(target, data)
    return dict(manifest=str(target), sha256=data['sha256'], files=len(files))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); print(json.dumps(build() if a.execute else dict(executed=False), indent=2))
