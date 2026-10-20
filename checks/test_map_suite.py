"""Map experiment protocol negative controls; no environments or model restores."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from project_paths import ROOT
from chapter3_bser.learning import map_suite as suite
from chapter3_bser.learning.map_contract import identity,research_config,metadata,validate_condition,validate_checkpoint_map,advance
from chapter3_bser.learning.map_report import summarize,paired_interval
from chapter3_bser.learning.train import Trainer
from chapter3_bser.learning.checkpoint import write_json,digest,state_digest
from chapter3_bser.learning.evaluation_records import finalize_evaluation,verified_rows
from chapter3_bser.d2.logging import JsonlJournal


def inputs():
    spec=suite.read(ROOT/'configs/map_comparison_v1.json')
    spec.pop('convergence',None)
    spec.update(seeds=[2729],train_episodes=1,max_steps=8,evaluation_count=1,checkpoint_interval=1,bootstrap_draws=16)
    scenes=suite.read(ROOT/'verification/mechanism_scenes.json')['scenarios']
    normal=copy.deepcopy(scenes[0]);normal['max_steps']=8
    collision=copy.deepcopy(scenes[3]);collision.update(max_steps=8,scenario_role='eval',scenario_split='eval',scenario_seed=910828,scenario_id='map_test_eval')
    return spec,{2729:dict(scenario_count=1,scenarios=[normal])},dict(scenario_count=1,scenarios=[collision])


def fixture(directory):
    spec,train,evaluation=inputs();root=Path(directory)/'collision_terminal/run'
    suite.prepare(root,spec,mechanism=True,training_manifests=train,evaluation_manifest=evaluation)
    return root,suite.load_plan(root)


def resign(root,plan):
    plan['assets']={p.relative_to(root).as_posix():suite.sha(p) for p in (root/'inputs').iterdir() if p.is_file()}
    plan['sha256']=digest({k:v for k,v in plan.items() if k!='sha256'});write_json(root/'plan.json',plan)


def synthetic_results(root,plan):
    scene=suite.read(root/'inputs/evaluation.json')['scenarios'][0]
    for job in plan['jobs']:
        config=suite.read(root/job['config']);method='D2' if job['seed'] is None else config['method']
        state={'test_tensor':torch.tensor([1. if job['map_id']=='OLD' else 2.])};policy=state_digest(state)
        if job['seed'] is not None:
            path=root/job['train'];path.mkdir(parents=True)
            summary=dict(episodes=1,optimizer_updates=1,actor_optimizer_steps=[1]*4,**metadata(config))
            write_json(path/'summary.json',summary);(path/'checkpoint_final.pt').write_bytes(b'synthetic metadata test only')
            row=dict(episode=1,global_step=8,team_discounted_return=1.,found_step=None,success=False,termination_reason='timeout')
            (path/'episodes.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8');(path/'training_metrics.jsonl').write_text('{}\n',encoding='utf-8')
            trainer=SimpleNamespace(initial_model_state_sha256='same_initial_'+job['learner'],model=SimpleNamespace(training_state_dict=lambda:state))
            suite._record_training(root,plan,job,trainer,summary)
        output=root/job['evaluation'];output.mkdir(parents=True)
        row=dict(scenario_id=scene['scenario_id'],scenario_seed=scene['scenario_seed'],scenario_sha256=digest(scene),environment_innovation_seed=scene['scenario_seed'],
            max_steps=8,gamma=.95,task_protocol=config['task_protocol'],reward_objective=config['reward_objective'],source_reward_revision=config['source_reward_revision'],
            waypoint_reward_timing=config['waypoint_reward_timing'],training_seed=job['seed'],method=method,policy_id=policy,source_review=plan['source_review'],
            trajectory_complete=True,found_step=None,found_penalized_steps=8,success=False,termination_reason='obstacle_collision',pre_found_collision=True,
            team_discounted_return=-2.,map_metrics=dict(runtime_step_seconds=.1,d2_fallbacks=0),**metadata(config))
        journal=JsonlJournal(output/'episodes.jsonl');journal.append(row)
        finalize_evaluation(journal,output,identity=plan['source_review'],manifest=suite.read(root/'inputs/evaluation.json'),policy_id=policy,method=method,
                            mechanism=True,**metadata(config))


class MapSuite(unittest.TestCase):
    def test_six_group_plan_has_four_independent_training_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder)
            self.assertEqual(len(plan['jobs']),6);self.assertEqual(sum(j['seed'] is not None for j in plan['jobs']),4)
            self.assertEqual({j['group'] for j in plan['jobs']},set(suite.GROUPS))

    def test_default_budget_is_eight_learned_policies(self):
        spec=suite.validate_spec(suite.read(ROOT/'configs/map_comparison_v1.json'))
        jobs=suite.jobs_for(spec);self.assertEqual(len(jobs),10);self.assertEqual(sum(j['seed'] is not None for j in jobs),8)
        self.assertEqual(spec['max_steps'],400);self.assertEqual(spec['train_episodes'],300)

    def test_old_is_default_off_and_new_is_explicit_core(self):
        spec,_,_=inputs()
        for name in ('OLD','NEW'):
            c=dict(map_experiment=dict(schema='auv.active.map_condition.v1',map_id=name,research_config=None if name=='OLD' else spec['new_map_config'],protocol_sha256='a'*64))
            self.assertEqual(identity(c)['map_id'],name)
            if name=='OLD':self.assertIsNone(research_config(c))
            else:self.assertTrue(research_config(c)['enabled'])

    def test_disabled_new_map_is_rejected(self):
        spec,_,_=inputs();spec['new_map_config']['enabled']=False
        with self.assertRaises(ValueError):suite.validate_spec(spec)

    def test_old_map_cannot_smuggle_a_research_estimator(self):
        c=dict(schema='auv.active.map_condition.v1',map_id='OLD',research_config={},protocol_sha256='a'*64)
        with self.assertRaises(ValueError):validate_condition(c)

    def test_checkpoint_native_map_cannot_be_relabelled(self):
        spec,_,_=inputs();c=dict(map_experiment=dict(schema='auv.active.map_condition.v1',map_id='NEW',research_config=spec['new_map_config'],protocol_sha256='a'*64))
        payload=metadata(c);validate_checkpoint_map(c,payload);payload['train_map']='OLD'
        with self.assertRaisesRegex(ValueError,'map identity'):validate_checkpoint_map(c,payload)

    def test_no_map_field_keeps_old_runtime_call_and_metadata(self):
        self.assertIsNone(research_config({}));self.assertEqual(metadata({}),{})
        calls=[];runtime=SimpleNamespace(config={},advance=lambda model,explore:calls.append((model,explore)) or 7)
        self.assertEqual(advance(runtime,'policy',explore=False),7);self.assertEqual(calls,[('policy',False)])
        self.assertFalse(hasattr(runtime,'_map_step_seconds'))

    def test_prepared_input_byte_drift_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder)
            with (root/plan['jobs'][0]['config']).open('a',encoding='utf-8') as stream:stream.write(' ')
            with self.assertRaisesRegex(ValueError,'input changed'):suite.load_plan(root)

    def test_resigned_plan_cannot_change_a_group_map(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);plan['jobs'][1]['map_id']='OLD';resign(root,plan)
            with self.assertRaisesRegex(ValueError,'job mismatch'):suite.load_plan(root)

    def test_resigned_config_cannot_change_learner_hyperparameters(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);path=root/plan['jobs'][2]['config'];c=suite.read(path);c['rl']['hidden_dim']=256;write_json(path,c);resign(root,plan)
            with self.assertRaisesRegex(ValueError,'learner contract'):suite.load_plan(root)

    def test_training_and_evaluation_physical_overlap_rejected(self):
        spec,train,_=inputs();scene=copy.deepcopy(train[2729]['scenarios'][0]);scene.update(scenario_id='new_label',scenario_seed=111,scenario_role='eval',scenario_split='eval')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'overlap'):
                suite.prepare(Path(folder)/'collision_terminal/run',spec,mechanism=True,training_manifests=train,evaluation_manifest=dict(scenarios=[scene]))

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);spec,train,evaluation=inputs()
            with self.assertRaises(FileExistsError):suite.prepare(root,spec,mechanism=True,training_manifests=train,evaluation_manifest=evaluation)

    def test_explicit_episode_order_does_not_resample_pool(self):
        trainer=Trainer.__new__(Trainer);trainer.config=dict(seed=2729,scenario_schedule=[1,0]);trainer.scenarios=[dict(id=0),dict(id=1)]
        self.assertEqual(trainer._scenario(1),(dict(id=1),2730));self.assertEqual(trainer._scenario(2),(dict(id=0),2731))

    def test_dry_run_does_not_read_spec_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()) as output:
            root=Path(folder)/'absent'
            suite.main(['prepare','--root',str(root),'--spec',str(root/'absent.json')])
            self.assertFalse(root.exists());self.assertFalse(json.loads(output.getvalue())['executed'])

    def test_formal_execution_requires_actual_linux(self):
        with patch.object(suite.platform,'system',return_value='Windows'):
            with self.assertRaisesRegex(ValueError,'actual Linux'):suite.require_linux_validation(None,{})

    def test_ten_episode_mechanism_plan_rejected_before_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder)
            with self.assertRaisesRegex(ValueError,'eight episodes'):suite.run(root,stage='all')

    def test_paired_zero_deltas_have_zero_intervals(self):
        ci=paired_interval(np.zeros((3,4)),draws=32,seed=4)
        self.assertEqual(ci['mean'],0);self.assertEqual(ci['lower'],0);self.assertEqual(ci['upper'],0)

    def test_bootstrap_is_local_reproducible_and_labels_one_seed(self):
        a=paired_interval([[1,2,3]],draws=32,seed=4);b=paired_interval([[1,2,3]],draws=32,seed=4)
        self.assertEqual(a,b);self.assertEqual(a['mean'],2);self.assertIn('scene uncertainty only',a['interval_scope'])

    def test_six_group_native_report_uses_seeds_and_paired_scenes(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);synthetic_results(root,plan);result=summarize(root,Path(folder)/'report')
            self.assertTrue(result['complete']);self.assertEqual(len(result['groups']),6);self.assertEqual(len(result['comparisons']),24)
            self.assertTrue((Path(folder)/'report/learning_curves.csv').is_file())

    def test_incomplete_evaluation_cannot_publish_complete_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);synthetic_results(root,plan)
            (root/plan['jobs'][-1]['evaluation']/'evaluation_identity.json').unlink()
            with self.assertRaises(OSError):summarize(root,Path(folder)/'report')
            self.assertFalse((Path(folder)/'report').exists())

    def test_resigned_rows_cannot_change_native_map_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);synthetic_results(root,plan);output=root/plan['jobs'][0]['evaluation']
            path=output/'episodes.jsonl';row=json.loads(path.read_text(encoding='utf-8'));row['eval_map']='NEW';path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            receipt=suite.read(output/'evaluation_identity.json');receipt['files'][path.name]=suite.sha(path);write_json(output/'evaluation_identity.json',receipt)
            with self.assertRaisesRegex(ValueError,'map identity'):verified_rows(path,plan['source_review'])

    def test_linux_script_is_lf_and_has_no_implicit_execute(self):
        raw=(ROOT/'scripts/linux_map_experiment.sh').read_bytes();self.assertNotIn(b'\r',raw)
        self.assertTrue(raw.startswith(b'#!/usr/bin/env bash\n'))
        self.assertNotIn(b'--execute "$@"',raw)

    def test_development_tasks_cannot_be_reused_as_formal_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);spec,train,evaluation=inputs()
            with self.assertRaisesRegex(ValueError,'overlap'):
                suite.prepare(Path(folder)/'collision_terminal/next',spec,mechanism=True,training_manifests=train,
                              evaluation_manifest=evaluation,exclude_runs=[root])

    def test_pilot_and_formal_use_separate_evaluation_seeds(self):
        a=suite.read(ROOT/'configs/map_comparison_v1.json');b=suite.read(ROOT/'configs/map_comparison_pilot_v1.json')
        self.assertNotEqual(a['evaluation_generator_seed'],b['evaluation_generator_seed'])

    def test_current_linux_validation_is_mandatory(self):
        with patch.object(suite.platform,'system',return_value='Linux'):
            with self.assertRaisesRegex(ValueError,'validation'):suite.require_linux_validation(None,{})

    def test_generated_id_namespace_preserves_physical_task_and_source_id(self):
        _,train,_=inputs();manifest=train[2729];value=suite.namespace_manifest(manifest,'train_gen123')
        self.assertNotEqual(value['scenarios'][0]['scenario_id'],manifest['scenarios'][0]['scenario_id'])
        self.assertEqual(value['scenarios'][0]['source_scenario_id'],manifest['scenarios'][0]['scenario_id'])
        from chapter3_bser.learning.config import scenario_fingerprint
        self.assertEqual(scenario_fingerprint(value['scenarios'][0]),scenario_fingerprint(manifest['scenarios'][0]))

    def test_distinct_development_manifest_is_copied_byte_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);spec,train,evaluation=inputs();scene=evaluation['scenarios'][0]
            scene['scenario_id']='distinct_eval';scene['scenario_seed']+=100
            scene['target_initial_position'][0]-=.1;scene['target_position'][0]-=.1
            second=Path(folder)/'collision_terminal/second'
            suite.prepare(second,spec,mechanism=True,training_manifests=train,evaluation_manifest=evaluation,exclude_runs=[root])
            current=suite.load_plan(second)
            self.assertEqual(len(current['excluded_manifests']),2)
            for item in current['excluded_manifests']:
                self.assertEqual(suite.sha(second/item['asset']),item['source_manifest_sha256'])
