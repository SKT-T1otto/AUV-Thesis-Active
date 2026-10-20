import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import random
import numpy as np
import torch
from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.checkpoint import validated_output
from chapter3_bser.learning.checkpoint import write_json
from chapter3_bser.d2.logging import JsonlJournal
from chapter3_bser.learning.metric_ordering import paired_ordering
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.learning.train import load_for_evaluation
from chapter3_bser.learning.scenarios import validate_manifest
from chapter3_bser.learning.evaluation_records import metric_row, finalize_evaluation
from chapter3_bser.learning.evaluation_records import verified_rows
from chapter3_bser.learning import map_contract

@contextmanager
def preserved_rng():
    """Evaluation must not reseed or consume the calling trainer's RNG."""
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

@preserved_rng()
def evaluate(checkpoint, manifest_path, output, *, mechanism=False):
    from chapter3_bser.learning.provenance import verify_reviewed_changes
    identity = verify_reviewed_changes()
    (model, config, payload) = load_for_evaluation(checkpoint, allow_mechanism=mechanism)
    return _evaluate_loaded(model,config,payload,manifest_path,output,identity=identity,mechanism=mechanism,checkpoint=checkpoint)


def _evaluate_loaded(model,config,payload,manifest_path,output,*,identity,mechanism=False,checkpoint=None):
    """Native evaluation kernel; verification can supply a fresh in-memory model."""
    from chapter3_bser.learning.checkpoint import state_digest
    from chapter3_bser.learning.provenance import require_review
    require_review(identity);map_contract.validate_checkpoint_map(config,payload)
    if state_digest(model.training_state_dict())!=payload['model_state_sha256']:raise ValueError('evaluation policy state identity mismatch')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    scenarios = validate_manifest(manifest, config, evaluation=True, mechanism=mechanism)
    ids = [s.get('scenario_id') for s in scenarios]
    if not scenarios or len(set(ids)) != len(ids) or set(ids) & set(payload['training_scenario_ids']):
        raise ValueError('evaluation requires unique held-out scenarios disjoint from training')
    for scene in scenarios:
        if scene.get('scenario_seed') in payload['training_scenario_seeds'] or scenario_fingerprint(scene) in payload['training_scenario_fingerprints']:
            raise ValueError('evaluation task seed or initial configuration overlaps training')
        if scene.get('scenario_role') not in ('eval', 'smoke_eval') or scene.get('scenario_split') != scene['scenario_role'] or scene.get('scenario_profile') != config['profile'] or (scene.get('max_steps') != config['max_steps']):
            raise ValueError('evaluation scenario role/profile/horizon mismatch')
    output = validated_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    journal = JsonlJournal(output / 'episodes.jsonl')
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for scene in scenarios:
            seed = int(scene['scenario_seed'])
            runtime = RepairedMissionRuntime(config, scene, seed=seed,map_research=map_contract.research_config(config))
            try:
                while not runtime.terminal:
                    map_contract.advance(runtime,model,explore=False)
                row = metric_row(runtime, config, scene, policy_id=payload['model_state_sha256'], method=config['method'], identity=identity, mechanism=mechanism)
                journal.append(row)
            finally:
                runtime.close()
        return finalize_evaluation(journal, output, identity=identity, manifest=manifest,
                                   policy_id=payload['model_state_sha256'], method=config['method'],
                                   mechanism=mechanism, checkpoint=None if checkpoint is None else str(Path(checkpoint).resolve()),
                                   **map_contract.metadata(config))
    finally:
        torch.set_num_threads(threads)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ev = commands.add_parser('evaluate')
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--manifest', required=True)
    ev.add_argument('--output', required=True)
    ev.add_argument('--execute', action='store_true')
    ev.add_argument('--mechanism-check', action='store_true')
    rank = commands.add_parser('ordering')
    rank.add_argument('--episodes', nargs='+', required=True)
    rank.add_argument('--output', required=True)
    rank.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'evaluate':
        if not args.execute:
            print(json.dumps(dict(executed=False, checkpoint=args.checkpoint, manifest=args.manifest)))
            return
        print(json.dumps(evaluate(args.checkpoint, args.manifest, args.output, mechanism=args.mechanism_check), indent=2))
    else:
        if not args.execute:
            print(json.dumps(dict(executed=False, inputs=args.episodes, output=args.output)))
            return
        from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review
        identity = verify_reviewed_changes()
        rows = []
        for path in args.episodes:
            rows.extend(verified_rows(path, identity)[0])
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        result = paired_ordering(rows)
        require_review(identity)
        write_json(destination, result)
        print(json.dumps(result, indent=2))
