"""Summarize verified D2/B2/B3 evaluations on the same held-out tasks."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from chapter3_bser.learning.checkpoint import write_json
from chapter3_bser.learning.evaluation_records import verified_rows
from chapter3_bser.learning.metric_ordering import paired_ordering
from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review

METHODS = ('D2', 'D2_B2_repair_v2', 'D2_B3_repair_v2')


def summarize(paths, output, *, expected=None):
    identity = verify_reviewed_changes()
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('unique completed evaluation files are required')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    all_rows, replicates, pairings, manifests, scopes = [], {}, [], set(), set()
    for path in paths:
        rows, receipt = verified_rows(path, identity)
        method = receipt['method']
        if method not in METHODS or any(r['method'] != method or r['policy_id'] != receipt['policy_id'] for r in rows):
            raise ValueError('evaluation method/policy mismatch')
        seed = rows[0]['training_seed']
        if any(r['training_seed'] != seed for r in rows) or (method, seed) in replicates:
            raise ValueError('one final policy per method/training seed is required')
        pairing = [(r['scenario_id'], r['scenario_seed'], r['scenario_sha256'], r['environment_innovation_seed'],
                    r['max_steps'], r['gamma'], r['task_protocol'], r['reward_objective'], r['source_reward_revision'], r['waypoint_reward_timing']) for r in rows]
        if pairings and pairing != pairings[0]:
            raise ValueError('evaluations must use the same ordered scenarios, innovations and task/reward conditions')
        pairings.append(pairing)
        manifests.add(receipt['manifest_sha256']); scopes.add(receipt['mechanism_only'])
        for row in rows:
            found = row['found_step']
            if found is not None and (type(found) is not int or not 0 <= found <= row['max_steps']):
                raise ValueError('invalid Found event time')
            if row['found_penalized_steps'] != (row['max_steps'] if found is None else found):
                raise ValueError('invalid failure-penalized Found metric')
        replicates[(method, seed)] = dict(found_rate=statistics.mean(r['found_step'] is not None for r in rows),
                                        pre_found_collision_rate=statistics.mean(bool(r['pre_found_collision']) for r in rows),
                                        found_penalized_steps=statistics.mean(r['found_penalized_steps'] for r in rows),
                                        success_rate=statistics.mean(bool(r['success']) for r in rows), scenarios=len(rows))
        all_rows.extend(rows)
    if len(manifests) != 1 or len(scopes) != 1:
        raise ValueError('different manifests or mechanism/research scopes cannot be merged')
    groups = []
    for method in METHODS:
        values = [r for (m, _), r in replicates.items() if m == method]
        if not values and (expected is None or method not in expected):
            continue
        wanted = len(values) if expected is None else expected.get(method, 0)
        group = dict(method=method, expected_replicates=wanted, completed_replicates=len(values), complete=len(values) == wanted)
        for metric in ('found_rate', 'pre_found_collision_rate', 'found_penalized_steps', 'success_rate'):
            data = [v[metric] for v in values]
            group[metric] = statistics.mean(data) if data else None
            group[metric + '_seed_sd'] = statistics.stdev(data) if len(data) > 1 else None
        groups.append(group)
    result = dict(schema='auv.active.summary.v1', source_review=identity, complete=all(g['complete'] for g in groups),
                  mechanism_only=next(iter(scopes)), scope='selected verified evaluation files' if expected is None else 'prepared suite',
                  groups=groups, performance_passed=False)
    ordering = paired_ordering(all_rows)
    require_review(identity)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', result); write_json(output / 'ordering.json', ordering)
    with (output / 'groups.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader(); writer.writerows(groups)
    lines = ['# D2 / D2+B2 / D2+B3', '', f"Complete: {result['complete']}", '',
             '| Method | Seeds complete | Found | Pre-Found collision | Penalized Found steps | Success |',
             '|---|---:|---:|---:|---:|---:|']
    for group in groups:
        metric = lambda key: 'pending' if group[key] is None else f'{group[key]:.4f}'
        lines.append(f"| {group['method']} | {group['completed_replicates']}/{group['expected_replicates']} | {metric('found_rate')} | {metric('pre_found_collision_rate')} | {metric('found_penalized_steps')} | {metric('success_rate')} |")
    lines.extend(['', '- Failure penalty is the configured horizon; collision ties count as pre-Found.',
                  '- D2 is evaluated once; shared D2 comparisons are not independent replicates.',
                  '- Seed SD describes training seeds on shared tasks; it is not a confidence interval.',
                  '- Missing jobs are pending, not failed missions. Mechanism data do not establish performance.',
                  '- Check actual episode/step costs and the original finish-started-episode budget policy.'])
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    result = summarize(args.episodes, args.output) if args.execute else dict(executed=False, inputs=args.episodes, output=args.output)
    print(json.dumps(result, indent=2)); return result
