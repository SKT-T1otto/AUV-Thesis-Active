"""Common D2/B2/B3 metrics and source-verified evaluation completion receipts."""
import hashlib
import json
from pathlib import Path
from chapter3_bser.learning.checkpoint import digest, write_json
from chapter3_bser.learning.provenance import require_review


def metric_row(runtime, config, scene, *, policy_id, method, identity, mechanism):
    row = runtime.summary()
    outcome = runtime.env.unwrapped.episode_outcome
    collision_step = outcome.first_collision_step
    collided = outcome.termination_reason == 'obstacle_collision'
    found = row['found_step']
    row.update(policy_id=policy_id, method=method, scenario_sha256=digest(scene), trajectory_complete=True,
               environment_innovation_seed=int(scene['scenario_seed']), max_steps=config['max_steps'],
               pre_found_collision=bool(collided and (found is None or collision_step <= found)),
               found_penalized_steps=config['max_steps'] if found is None else found,
               source_review=identity, training_seed=None if method == 'D2' else config['seed'],
               mechanism_only=mechanism, performance_passed=False)
    return row


def finalize_evaluation(journal, output, *, identity, manifest, policy_id, method, mechanism, checkpoint=None):
    # Raw journals are diagnostic when the final source gate fails. Consumers
    # require the verified receipt and never infer success from raw rows alone.
    require_review(identity)
    journal.materialize()
    output = Path(output)
    files = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ('episodes.jsonl', 'episodes.json')}
    receipt = dict(schema='auv.active.evaluation.v1', status='completed', completed=len(journal), source_review=identity,
                   manifest_sha256=digest(manifest), policy_id=policy_id, method=method, checkpoint=checkpoint,
                   mechanism_only=mechanism, performance_passed=False, files=files)
    require_review(identity)
    write_json(output / 'evaluation_identity.json', receipt)
    return receipt


def verified_rows(path, identity):
    path = Path(path).resolve()
    receipt = json.loads((path.parent / 'evaluation_identity.json').read_text(encoding='utf-8'))
    if receipt.get('schema') != 'auv.active.evaluation.v1' or receipt.get('status') != 'completed' or receipt.get('source_review') != identity:
        raise ValueError('evaluation source/completion identity mismatch')
    if path.name not in receipt.get('files', {}) or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['files'][path.name]:
        raise ValueError('evaluation rows changed or are not committed by the receipt')
    text = path.read_text(encoding='utf-8')
    rows = json.loads(text) if path.suffix == '.json' else [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(rows) != receipt['completed'] or not rows or any(r.get('source_review') != identity or r.get('trajectory_complete') is not True for r in rows):
        raise ValueError('evaluation row count/source/completeness mismatch')
    return rows, receipt
