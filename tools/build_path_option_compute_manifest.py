"""Freeze the explicitly reviewed compute-only evolution; never rewrite v1-v3."""
import argparse
import json
from project_paths import ROOT
from chapter3_bser.path_option.source import sha, digest, legacy_digest, write

CHANGED = {
    'chapter3_bser/path_option/networks.py': 'Expose call-local policy and critic heads without changing modules or parameters',
    'chapter3_bser/path_option/learner.py': 'Reuse no-grad encodings and post-critic values; preserve actor gradient graphs',
    'chapter3_bser/path_option/source.py': 'Bind current code to new v4 compute identity; keep old manifests and evidence frozen',
    'tools/validate_path_option.py': 'Include independent frozen-parent compute equivalence in current validation',
}
ADDED = [
    'checks/test_path_option_compute.py', 'tools/benchmark_path_option_compute.py',
    'tools/build_path_option_compute_manifest.py', 'docs/PATH_OPTION_COMPUTE_20261112.md',
    'verification/path_option_compute_v1/20261112_v1/before.json',
    *['verification/path_option_compute_v1/20261112_v1/before/chapter3_bser/path_option/'+n+'.py'
      for n in ('learner', 'networks', 'source')],
]


def build():
    target = ROOT/'provenance/path_option_manifest_v4.json'
    evolution = ROOT/'provenance/path_option_compute_evolution_v1.json'
    if target.exists() or evolution.exists():
        raise FileExistsError('compute manifest/evolution already frozen')
    parent_name = 'provenance/path_option_manifest_v3.json'
    parent = json.loads((ROOT/parent_name).read_text(encoding='utf-8'))
    if (parent['sha256'] != 'cf81d42a92a46377d520f5e244a0cb424bc81cc63ead9fc819014c403db552ff'
            or digest({k: v for k, v in parent.items() if k != 'sha256'}) != parent['sha256']):
        raise ValueError('frozen v3 parent changed')
    files = dict(parent['files'])
    for name, expected in files.items():
        actual = legacy_digest(ROOT/name) if name in parent['normalized_existing_dispatch'] else sha(ROOT/name)
        if name not in CHANGED and actual != expected:
            raise ValueError('unreviewed parent change: '+name)
    for name in (*CHANGED, *ADDED, parent_name):
        files[name] = sha(ROOT/name)
    changes = [dict(path=name, before_sha256=parent['files'].get(name), after_sha256=files[name],
        reason=CHANGED.get(name, 'Bounded compute comparison, frozen reference, reproduction or documentation'))
        for name in sorted(files) if files[name] != parent['files'].get(name)]
    write(evolution, dict(schema='auv.path_option.compute_evolution.v1', parent_source=parent['sha256'],
        parent_manifest_sha256=sha(ROOT/parent_name), changes=changes,
        algorithm_reward_training_configuration_changed=False,
        preexisting_pure_marl_dispatch_untouched=True,
        evidence_note='Outputs/losses/gradients/Adam/RNG compared to frozen v3; native validation is separate'))
    files[evolution.relative_to(ROOT).as_posix()] = sha(evolution)
    data = dict(schema=parent['schema'], parent_commit='27a4053af78887941aa888227a19b991eef9ed8d',
        normalized_existing_dispatch=parent['normalized_existing_dispatch'], files=files,
        parent_source_sha256=parent['sha256'], compute_revision='call_local_reuse_v1',
        parent_algorithm_bytes_changed=True, algorithm_configuration_changed=False,
        automatic_native_budget_remaining=0, user_untracked_results_excluded=True,
        historical_evidence_retained=True, old_weights_relabeling_allowed=False)
    data['sha256'] = digest(data)
    write(target, data)
    return dict(manifest=str(target), sha256=data['sha256'], files=len(files), changed_parent=list(CHANGED))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    print(json.dumps(build() if a.execute else dict(executed=False), indent=2))


if __name__ == '__main__':
    main()
