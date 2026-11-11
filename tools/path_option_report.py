"""Incomplete jobs remain visible; scenario pairs are not training replicates."""
import csv
import json
from pathlib import Path
import numpy as np
from chapter3_bser.path_option.source import write, sha
from chapter3_bser.path_option.metrics import learning_assessment


def read_lines(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b'\n')+1
    return [json.loads(x) for x in data[:end].splitlines() if x]


def build_report(root, out, plan):
    out.mkdir(parents=True)
    jobs = []; curves = []
    for arm in plan['primary_arms']:
        for seed in plan['seeds']:
            job = root/'jobs'/f'{arm}_seed{seed}'
            complete = job/'completed.json'
            curve = read_lines(job/'development.jsonl')
            row = dict(arm=arm, seed=seed, completed=complete.exists(), evaluation_points=len(curve),
                assessment=learning_assessment(curve))
            if complete.exists():
                receipt = json.loads(complete.read_text(encoding='utf-8'))
                row.update(episodes=receipt['episodes'], physical_steps=receipt['physical_steps'], updates=receipt['updates'],
                    receipt_sha256=sha(complete))
            jobs.append(row)
            curves.extend(dict(arm=arm, seed=seed, **x) for x in curve)
    evaluations = []; incomplete_evaluations = []
    for request in sorted(root.rglob('evaluation_request.json')):
        if not (request.parent/'evaluation.json').exists():
            incomplete_evaluations.append(dict(folder=str(request.parent.relative_to(root)),
                request=json.loads(request.read_text(encoding='utf-8')),
                committed_episodes=len(read_lines(request.parent/'metrics.jsonl')),
                status='incomplete; no performance comparison accepted'))
    for meta in sorted(root.rglob('evaluation.json')):
        folder = meta.parent
        rows = read_lines(folder/'metrics.jsonl')
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        record = json.loads(meta.read_text(encoding='utf-8'))
        if len(rows) != summary['episodes']:
            raise ValueError('incomplete evaluation journal')
        evaluations.append(dict(folder=str(folder.relative_to(root)), metadata=record,
            summary=summary, metrics_sha256=sha(folder/'metrics.jsonl')))
    # Pair only completed evaluations with identical suite and intervention.
    deltas = []
    for x in range(len(evaluations)):
        for y in range(x):
            a, b = evaluations[x], evaluations[y]
            if (a['metadata']['suite'], a['metadata']['intervention']) != (b['metadata']['suite'], b['metadata']['intervention']):
                continue
            ra = {r['scenario_id']: r for r in read_lines(root/a['folder']/'metrics.jsonl')}
            rb = {r['scenario_id']: r for r in read_lines(root/b['folder']/'metrics.jsonl')}
            if set(ra) != set(rb):
                continue
            deltas.append(dict(a=a['folder'], b=b['folder'], scenes=len(ra),
                success_delta=float(np.mean([ra[k]['success']-rb[k]['success'] for k in ra])),
                collision_delta=float(np.mean([ra[k]['collision']-rb[k]['collision'] for k in ra])),
                penalized_steps_delta=float(np.mean([ra[k]['penalized_steps']-rb[k]['penalized_steps'] for k in ra])),
                inference='paired scene differences for these policies, not independent training replicates'))
    seed_groups = {}
    for e in evaluations:
        m = e['metadata']
        key = '|'.join(str(m[k]) for k in ('mode', 'suite', 'retrained_variant', 'intervention'))
        seed = m.get('training_seed')
        group = seed_groups.setdefault(key, {})
        if seed in group:
            raise ValueError('duplicate evaluation of the same trained seed; select one frozen evaluation per condition')
        group[seed] = e['summary']
    across_seeds = {}
    for key, group in seed_groups.items():
        values = list(group.values())
        across_seeds[key] = dict(training_replicates=len(values) if list(group) != [None] else 0,
            per_seed=group,
            mean={k: float(np.mean([v[k] for v in values])) for k in ('success', 'collision', 'timeout', 'penalized_steps')},
            seed_sd={k: float(np.std([v[k] for v in values], ddof=1)) if len(values) > 1 else None for k in ('success', 'collision', 'timeout', 'penalized_steps')},
            note='seed variability reported separately; no scene-count-as-seed confidence interval')
    result = dict(jobs=jobs, evaluations=evaluations, incomplete_evaluations=incomplete_evaluations,
        paired_deltas=deltas, across_training_seeds=across_seeds,
        complete_training_jobs=sum(x['completed'] for x in jobs), expected_training_jobs=len(jobs),
        performance_accepted=False, final_test_used=bool(evaluations), convergence_proved=False)
    write(out/'report.json', result)
    if curves:
        keys = ['arm', 'seed', 'episode', 'physical_steps', 'updates', 'success', 'collision', 'timeout', 'penalized_steps', 'team_discounted_return', 'forced_fraction']
        with (out/'learning_curves.csv').open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(curves)
        # Standalone SVG: no plotting dependency, no smoothing or best-seed selection.
        colors = {'A': '#165DCC', 'B': '#BD5D10', 'C': '#168354', 'D': '#A63D82'}
        paths = []
        for arm in plan['primary_arms']:
            for seed in plan['seeds']:
                selected = [x for x in curves if x['arm'] == arm and x['seed'] == seed]
                points = ' '.join(f'{60+680*x["episode"]/plan["episodes"]:.2f},{350-300*x["success"]:.2f}' for x in selected)
                paths.append(f'<polyline fill="none" stroke="{colors.get(arm, "black")}" points="{points}"/>')
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="410"><rect width="800" height="410" fill="white"/><path d="M60 40V350H750" fill="none" stroke="black"/><text x="60" y="25">Development success; each line is a training seed</text>'+''.join(paths)+'<text x="60" y="390">Episode (0 to '+str(plan['episodes'])+'); success (0 to 1)</text></svg>'
        (out/'success.svg').write_text(svg, encoding='utf-8')
    return dict(report=str(out/'report.json'), complete_jobs=result['complete_training_jobs'], expected_jobs=len(jobs))
