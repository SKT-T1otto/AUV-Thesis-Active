"""Current manifest, evaluation, source and manual-entry contracts."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from project_paths import ROOT
from chapter3_bser.d2.logging import JsonlJournal
from chapter3_bser.learning.checkpoint import write_json, digest
from chapter3_bser.learning.config import validate
from chapter3_bser.learning.scenarios import validate_manifest, export
from chapter3_bser.learning.evaluation_records import metric_row, finalize_evaluation, verified_rows
from chapter3_bser.learning.provenance import verify_reviewed_changes, research_readiness


def scene(index=0):
    value = copy.deepcopy(json.loads((ROOT / 'verification/mechanism_scenes.json').read_text())['scenarios'][-1])
    value.update(scenario_id=f'workflow_eval_{index}', scenario_seed=810000 + index, scenario_role='eval', scenario_split='eval')
    value['target_initial_position'][0] -= .01 * index
    value['target_position'][0] -= .01 * index
    return value


def config():
    return validate(json.loads((ROOT / 'configs/b2_mechanism.json').read_text()))


class Workflow(unittest.TestCase):
    def test_01_explicit_role_export_preserves_physical_values(self):
        generated = dict(scenarios=[scene()], scenario_count=1)
        generated['scenarios'][0].update(scenario_role='validation', scenario_split='validation')
        with tempfile.TemporaryDirectory() as directory, patch('chapter3_bser.learning.scenarios.build_scenario_manifests', return_value={'M20_MOVING_UNKNOWN_MULTI': generated}) as sampler:
            path = Path(directory) / 'eval.json'
            export(path, split='eval', count=1, seed=92729, max_steps=8)
            actual = json.loads(path.read_text())
            self.assertEqual(sampler.call_args.kwargs['split'], 'validation')
            self.assertTrue(actual['role_mapping_explicit'])
            expected = copy.deepcopy(generated['scenarios'][0]); expected.update(scenario_role='eval', scenario_split='eval', max_steps=8)
            self.assertEqual(actual['scenarios'][0], expected)

    def test_02_mechanism_nine_scene_rejected_before_runtime(self):
        manifest = dict(scenarios=[scene(i) for i in range(9)])
        with self.assertRaisesRegex(ValueError, 'mechanism evaluation'):
            validate_manifest(manifest, config(), evaluation=True, mechanism=True)

    def test_03_research_evaluation_does_not_inherit_mechanism_count_limit(self):
        self.assertEqual(len(validate_manifest(dict(scenarios=[scene(i) for i in range(9)]), config(), evaluation=True)), 9)

    def test_04_eval_cannot_accept_train_role(self):
        value = scene(); value.update(scenario_role='train', scenario_split='train')
        with self.assertRaises(ValueError):validate_manifest(dict(scenarios=[value]), config(), evaluation=True)

    def test_05_duplicate_physical_tasks_rejected_after_id_and_seed_change(self):
        first = scene(); second = copy.deepcopy(first); second.update(scenario_id='renamed', scenario_seed=999999)
        with self.assertRaisesRegex(ValueError, 'physical'):
            validate_manifest(dict(scenarios=[first, second]), config(), evaluation=True)

    def test_06_d2_metric_row_is_directly_accepted_by_ordering(self):
        from chapter3_bser.learning.metric_ordering import paired_ordering
        identity = verify_reviewed_changes()
        fake = SimpleNamespace(summary=lambda: dict(scenario_id=scene()['scenario_id'], scenario_seed=scene()['scenario_seed'], found_step=1, team_discounted_return=1., success=False,
            gamma=.95, task_protocol='collision_terminal_v1', reward_objective='team_mean_v1',
            source_reward_revision='execution_reward_v2_after_terminal_override_v1', waypoint_reward_timing='arrival_before_reward_v1'),
            env=SimpleNamespace(unwrapped=SimpleNamespace(episode_outcome=SimpleNamespace(first_collision_step=1, termination_reason='obstacle_collision'))))
        row = metric_row(fake, config(), scene(), policy_id='zero', method='D2', identity=identity, mechanism=True)
        self.assertTrue(row['pre_found_collision']); self.assertIsNone(row['training_seed'])
        self.assertEqual(paired_ordering([row])['paired_scenario_groups'], 0)

    def test_07_final_source_failure_cannot_write_completion_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); journal = JsonlJournal(output / 'episodes.jsonl'); journal.append(dict(diagnostic=True))
            with patch('chapter3_bser.learning.evaluation_records.require_review', side_effect=ValueError('injected source rejection')):
                with self.assertRaisesRegex(ValueError, 'source rejection'):
                    finalize_evaluation(journal, output, identity={}, manifest={}, policy_id='test', method='D2', mechanism=True)
            self.assertFalse((output / 'evaluation_identity.json').exists())
            self.assertFalse((output / 'episodes.json').exists())

    def test_08_tampered_evaluation_rows_rejected(self):
        identity = verify_reviewed_changes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); journal = JsonlJournal(output / 'episodes.jsonl')
            journal.append(dict(source_review=identity, trajectory_complete=True))
            finalize_evaluation(journal, output, identity=identity, manifest={}, policy_id='test', method='D2', mechanism=True)
            verified_rows(output / 'episodes.jsonl', identity)
            with (output / 'episodes.jsonl').open('a') as stream:stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'changed'):verified_rows(output / 'episodes.jsonl', identity)

    def test_09_missing_readiness_never_admits_research(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):research_readiness(verify_reviewed_changes(), root=directory)

    def test_10_prepared_plan_inputs_cannot_drift(self):
        from chapter3_bser.learning.suite import prepare, load_plan
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory); manifest = parent / 'eval.json'; write_json(manifest, dict(scenarios=[scene()]))
            root = parent / 'collision_terminal/run'
            prepare(root, manifest, seeds=[2729], mechanism=True)
            self.assertEqual(len(load_plan(root)['jobs']), 3)
            with (root / 'inputs/D2_B2_seed2729.json').open('a') as stream:stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'input changed'):load_plan(root)

    def test_11_suite_lock_refuses_a_second_owner_and_removes_own_lock(self):
        from chapter3_bser.learning.suite import run_lock
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.active_suite.lock'
            with run_lock(directory):
                with self.assertRaises(FileExistsError):
                    with run_lock(directory):pass
                self.assertTrue(path.exists())
            self.assertFalse(path.exists())

    def test_12_specific_command_help_reaches_its_parser(self):
        import cli
        with patch('sys.argv', ['cli', 'scenarios', '--help']), contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error:cli.main()
        self.assertEqual(error.exception.code, 0); self.assertIn('--count', output.getvalue())

    def test_13_missing_execute_does_not_create_output_or_restore_model(self):
        import cli
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'never_created')
            commands = [
                ['train', '--config', str(ROOT/'configs/b2.json'), '--output', path],
                ['reference', '--config', str(ROOT/'configs/b2.json'), '--manifest', 'missing', '--output', path],
                ['evaluate', '--checkpoint', 'missing', '--manifest', 'missing', '--output', path],
                ['summarize', '--episodes', 'missing', '--output', path],
                ['ordering', '--episodes', 'missing', '--output', path],
                ['suite', 'prepare', '--root', path, '--manifest', 'missing'],
                ['scenarios', '--output', path]]
            for command in commands:
                with self.subTest(command=command[0]), patch('sys.argv', ['cli', *command]), contextlib.redirect_stdout(io.StringIO()) as output:
                    cli.main()
                    self.assertFalse(json.loads(output.getvalue())['executed'])
                    self.assertFalse(Path(path).exists())

    def test_14_old_distribution_checkpoint_rejected_before_model_construction(self):
        from chapter3_bser.learning.train import load_for_evaluation
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'; c = config()
            torch.save(dict(schema=c['schema'], config=c, config_hash=digest(c), source_review=verify_reviewed_changes(), artifact_schema='old'), path)
            with patch('chapter3_bser.learning.train.build_model', side_effect=AssertionError('must not construct')):
                with self.assertRaises(ValueError):load_for_evaluation(path, allow_mechanism=True)
