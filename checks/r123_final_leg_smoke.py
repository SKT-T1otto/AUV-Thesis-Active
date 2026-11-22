"""Two short original search prefixes and one near-field interface fixture, each paired."""
import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np

from tools.r123_expansion import DATA
from tools.r123_final_leg import compare, identity, match_prefix, measures, simulate
from tools.r123_task_sync_experiment import fresh, read, require, write
from chapter3_bser.path_option.source import sha


def run(output, part):
    source = identity()
    out = fresh(output)
    scenarios = read(DATA / 'scenes.json')['scenarios']
    options = read(DATA / 'options.json')
    records = []
    cases = {'search': ((13, 8), (17, 8)), 'near': ((22, 24),)}
    for index, cap in (cases['search'] + cases['near'] if part == 'all' else cases[part]):
        near_fixture = index == 22
        config = dict(read(DATA / 'config.json'), max_steps=cap)
        scene = dict(scenarios[index], max_steps=cap)
        if near_fixture:
            scene['scenario_id'] = 'final_leg_interface_fixture'
            positions = list(scene['initial_agent_positions'])
            positions[1] = [17.95, 16.75, 6.75]
            positions[3] = [14.8, 15.3, 5.85]
            require(all(np.linalg.norm(np.subtract(a, b)) >= 2.8 for a, b in combinations(positions, 2)),
                    'fixture violates the original minimum separation')
            scene['initial_agent_positions'] = positions
        runs = []
        for direct in (False, True):
            raw, elapsed = simulate(config, scene, options, index, 20261119 + index, direct)
            measured = measures(raw, scene, elapsed)
            path = out / f'{index:04d}_{"direct" if direct else "grid"}.json'
            write(path, raw)
            records.append(dict(scene=scene['scenario_id'], direct=direct, cap=cap, measures=measured,
                                raw_file=path.name, raw_sha256=sha(path)))
            runs.append(raw)
        prefix = compare(*runs, measured)
        if near_fixture:
            require(measured['direct_actions'] > 0 and measured['handoff_step'] is not None,
                    'fixture did not exercise the new connection')
        else:
            require(prefix == cap and not measured['direct_actions'], 'original search prefix changed')
            match_prefix(*runs, cap)
    result = dict(source=source, part=part, records=records, environment_initializations=len(records),
                  physical_steps=sum(r['measures']['steps'] for r in records), passed=True,
                  training_updates=0, checkpoint_load=False)
    require(identity() == source and result['physical_steps'] <= 80, 'smoke cap/source changed')
    write(out / 'completed.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--part', choices=('all', 'search', 'near'), default='all')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.execute:
        run(args.output, args.part)
