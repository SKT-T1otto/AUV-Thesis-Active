"""Evaluate the D2 prior with exactly zero normalized RL residuals."""
import argparse
import json
from pathlib import Path
import torch
from chapter3_bser.learning.config import validate
from chapter3_bser.learning.provenance import training_admission
from chapter3_bser.learning.checkpoint import validated_output, digest
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.learning.scenarios import validate_manifest
from chapter3_bser.learning.evaluation_records import metric_row, finalize_evaluation
from chapter3_bser.learning.evaluate import preserved_rng
from chapter3_bser.d2.logging import JsonlJournal


class ZeroResidualPolicy:
    def step(self, observations, explore=False):
        return [torch.zeros(1, 3) for _ in range(4)]


@preserved_rng()
def evaluate_reference(config, manifest_path, output, *, mechanism=False):
    config = validate(config)
    identity = training_admission(mechanism=mechanism, config=config)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    scenes = validate_manifest(manifest, config, evaluation=True, mechanism=mechanism)
    condition_keys = ('profile', 'max_steps', 'planner_protocol', 'repairs', 'waypoint_reward_timing', 'task_protocol', 'reward', 'reward_objective')
    policy_id = digest(dict(method='D2', source=identity, action='zero_normalized_residual', conditions={k: config[k] for k in condition_keys}))
    output = validated_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    journal = JsonlJournal(output / 'episodes.jsonl')
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for scene in scenes:
            runtime = RepairedMissionRuntime(config, scene, seed=int(scene['scenario_seed']))
            try:
                while not runtime.terminal:
                    runtime.advance(ZeroResidualPolicy(), explore=False)
                journal.append(metric_row(runtime, config, scene, policy_id=policy_id, method='D2', identity=identity, mechanism=mechanism))
            finally:
                runtime.close()
        return finalize_evaluation(journal, output, identity=identity, manifest=manifest, policy_id=policy_id, method='D2', mechanism=mechanism)
    finally:
        torch.set_num_threads(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mechanism-check', action='store_true')
    args = parser.parse_args(argv)
    config = validate(json.loads(Path(args.config).read_text(encoding='utf-8')))
    if not args.execute:
        result = dict(executed=False, method='D2', manifest=args.manifest, output=args.output)
    else:
        result = evaluate_reference(config, args.manifest, args.output, mechanism=args.mechanism_check)
    print(json.dumps(result, indent=2)); return result
