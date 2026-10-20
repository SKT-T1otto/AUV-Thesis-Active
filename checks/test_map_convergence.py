"""Fixed validation, checkpoint integrity and plateau negative controls; no rollouts."""
import copy
import json
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import numpy as np
from project_paths import ROOT
from chapter3_bser.learning import map_suite as suite
from chapter3_bser.learning import map_convergence as cv
from chapter3_bser.learning.map_contract import metadata
from chapter3_bser.learning.map_report import summarize
from chapter3_bser.learning.checkpoint import write_json,digest,state_digest
from chapter3_bser.learning.evaluation_records import finalize_evaluation
from chapter3_bser.d2.logging import JsonlJournal


def fixture(folder):
    from checks.test_map_suite import inputs
    spec,training,evaluation=inputs()
    settings=suite.read(ROOT/'configs/map_comparison_v1.json')['convergence']
    settings.update(validation_count=2)
    spec.update(train_episodes=3,convergence=settings)
    original=training[2729]['scenarios'][0];scenes=[]
    for i in range(3):
        scene=copy.deepcopy(original);scene.update(scenario_id='train_'+str(i),scenario_seed=910100+i)
        scene['target_initial_position'][0]+=.01*i;scene['target_position'][0]+=.01*i;scenes.append(scene)
    training[2729]=dict(scenario_count=3,scenarios=scenes)
    validation=copy.deepcopy(evaluation);validation['scenarios']=[]
    for i in range(2):
        scene=copy.deepcopy(evaluation['scenarios'][0]);scene.update(scenario_id='validation_'+str(i),scenario_seed=911100+i)
        scene['target_initial_position'][0]-=.1*(i+1);scene['target_position'][0]-=.1*(i+1);validation['scenarios'].append(scene)
    validation['scenario_count']=2
    root=Path(folder)/'collision_terminal/convergence'
    suite.prepare(root,spec,mechanism=True,training_manifests=training,evaluation_manifest=evaluation,validation_manifest=validation)
    return root,suite.load_plan(root)


def synthetic_training(root,plan):
    for job in plan['jobs']:
        if job['seed'] is None:continue
        config=suite.read(root/job['config']);path=root/job['train'];path.mkdir(parents=True)
        state=dict(test_tensor=torch.tensor([float(job['map_id']=='NEW')]))
        summary=dict(episodes=3,optimizer_updates=1,actor_optimizer_steps=[1]*4,**metadata(config))
        write_json(path/'summary.json',summary)
        for name in cv.checkpoint_files(plan['spec']):(path/name).write_bytes(('synthetic '+name).encode())
        rows=[dict(episode=i,global_step=i*8,team_discounted_return=1.,found_step=None,success=False,termination_reason='timeout') for i in range(1,4)]
        (path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        (path/'training_metrics.jsonl').write_text(json.dumps(dict(global_step=8,update=1,agents=[dict(actor_updated=True,actor_loss=.1,critic_loss=.2)]*4))+'\n',encoding='utf-8')
        trainer=SimpleNamespace(initial_model_state_sha256='a'*64,model=SimpleNamespace(training_state_dict=lambda:state))
        suite._record_training(root,plan,job,trainer,summary)


def evaluation_rows(root,plan,job,manifest,output,policy,episode):
    config=suite.read(root/job['config']);method='D2' if job['seed'] is None else config['method']
    output.mkdir(parents=True);journal=JsonlJournal(output/'episodes.jsonl')
    for scene in manifest['scenarios']:
        row=dict(scenario_id=scene['scenario_id'],scenario_seed=scene['scenario_seed'],scenario_sha256=digest(scene),environment_innovation_seed=scene['scenario_seed'],
            max_steps=8,gamma=.95,task_protocol=config['task_protocol'],reward_objective=config['reward_objective'],source_reward_revision=config['source_reward_revision'],
            waypoint_reward_timing=config['waypoint_reward_timing'],training_seed=job['seed'],method=method,policy_id=policy,source_review=plan['source_review'],
            trajectory_complete=True,found_step=None if episode==0 else 2,found_penalized_steps=8 if episode==0 else 2,success=episode!=0,
            termination_reason='timeout',pre_found_collision=False,team_discounted_return=1. if episode==0 else 2.,
            map_metrics=dict(runtime_step_seconds=.1,d2_fallbacks=0),**metadata(config))
        journal.append(row)
    finalize_evaluation(journal,output,identity=plan['source_review'],manifest=manifest,policy_id=policy,method=method,mechanism=True,**metadata(config))


def fake_point(root,plan,job,episode,trained):
    policy=trained['initial_model_state_sha256'] if episode==0 else trained['model_state_sha256'] if episode==3 else digest(dict(job=job['name'],episode=episode))
    evaluation_rows(root,plan,job,suite.read(root/'inputs/validation.json'),root/'jobs'/job['name']/'diagnostics'/f'episode_{episode:06d}',policy,episode)
    return policy


def curve(values):
    return [dict(episode=i*50,success_rate=v,found_rate=v,collision_rate=0.,found_penalized_steps=400-200*v,team_discounted_return=10.+10*v) for i,v in enumerate(values)]


class MapConvergence(unittest.TestCase):
    def settings(self):return suite.read(ROOT/'configs/map_comparison_v1.json')['convergence']

    def test_default_budget_and_all_seven_checkpoints(self):
        spec=suite.validate_spec(suite.read(ROOT/'configs/map_comparison_v1.json'))
        self.assertEqual(spec['seeds'],[2729,3729]);self.assertEqual(spec['train_episodes'],300)
        self.assertEqual(cv.points(spec),[0,50,100,150,200,250,300])
        self.assertEqual(len(cv.checkpoint_files(spec)),6)
        self.assertEqual(8*300,2400);self.assertEqual(8*7*spec['convergence']['validation_count'],672)

    def test_invalid_tolerances_are_rejected(self):
        for value in (0,True,float('nan'),2):
            c=self.settings();c['rate_tolerance']=value
            with self.assertRaises(ValueError):cv.validate_settings(c)

    def test_too_few_checkpoints_rejected_before_preparing(self):
        spec=suite.read(ROOT/'configs/map_comparison_v1.json');spec['checkpoint_interval']=300
        with self.assertRaisesRegex(ValueError,'enough saved'):suite.validate_spec(spec)

    def test_initial_policy_reconstruction_matches_trainer_and_preserves_rng(self):
        from chapter3_bser.learning.train import Trainer
        from chapter3_bser.learning.evaluate import preserved_rng
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder)
            with preserved_rng():
                for learner in ('B2','B3'):
                    job=next(j for j in plan['jobs'] if j['learner']==learner)
                    trainer=Trainer(suite.read(root/job['config']),root/job['train'],mechanism=True)
                    trained=dict(initial_model_state_sha256=trainer.initial_model_state_sha256)
                    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
                    def kernel(model,config,payload,*args,**kwargs):
                        self.assertEqual(state_digest(model.training_state_dict()),trainer.initial_model_state_sha256)
                        random.random();np.random.random();torch.rand(1)
                    with patch('chapter3_bser.learning.evaluate._evaluate_loaded',side_effect=kernel),patch('chapter3_bser.learning.train.load_for_evaluation',side_effect=AssertionError('no file restore for initial point')):
                        self.assertEqual(cv.evaluate_point(root,plan,job,0,trained),trainer.initial_model_state_sha256)
                    self.assertEqual(random.getstate(),before[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],before[1][1]));self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))

    def test_wrong_checkpoint_episode_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);job=next(j for j in plan['jobs'] if j['seed'] is not None);config=suite.read(root/job['config'])
            with patch('chapter3_bser.learning.train.load_for_evaluation',return_value=(None,config,dict(episodes=2))),patch('chapter3_bser.learning.evaluate._evaluate_loaded',side_effect=AssertionError('must reject first')):
                with self.assertRaisesRegex(ValueError,'episode/config'):cv.evaluate_point(root,plan,job,1,dict(initial_model_state_sha256='a'*64))

    def test_flat_bad_policy_is_not_called_converged(self):
        result=cv.assess(curve([0]*7),self.settings(),400)
        self.assertEqual(result['status'],'stable_without_clear_task_gain');self.assertFalse(result['provisional_plateau'])

    def test_improved_stable_policy_is_only_provisional(self):
        result=cv.assess(curve([0,.1,.3,.5,.5,.5,.5]),self.settings(),400)
        self.assertTrue(result['provisional_plateau']);self.assertIn('not',result['interpretation'])

    def test_rising_curve_is_still_changing(self):
        self.assertEqual(cv.assess(curve([0,.1,.2,.3,.4,.6,.8]),self.settings(),400)['status'],'still_changing')

    def test_return_alone_does_not_establish_task_gain(self):
        rows=curve([0]*7)
        for r in rows[1:]:r['team_discounted_return']=20.
        self.assertEqual(cv.assess(rows,self.settings(),400)['status'],'stable_without_clear_task_gain')

    def test_collision_regression_blocks_plateau(self):
        rows=curve([0,.5,.5,.5,.5,.5,.5])
        for r in rows[1:]:r['collision_rate']=.5
        self.assertEqual(cv.assess(rows,self.settings(),400)['status'],'stable_with_collision_regression')

    def test_missing_updates_and_nonfinite_curve_rejected(self):
        self.assertEqual(cv.assess(curve([0,.5,.5,.5]),self.settings(),400,learned=False)['status'],'no_learning_updates')
        rows=curve([0,.5,.5,.5]);rows[-1]['team_discounted_return']=float('nan')
        with self.assertRaises(ValueError):cv.assess(rows,self.settings(),400)

    def test_validation_is_frozen_and_disjoint_from_train_and_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);validation=suite.read(root/'inputs/validation.json')['scenarios']
            suite.disjoint(validation,suite.read(root/'inputs/evaluation.json')['scenarios'])
            suite.disjoint(validation,suite.read(root/'inputs/train_seed2729.json')['scenarios'])
            self.assertIn('inputs/validation.json',plan['assets'])

    def test_resigned_validation_overlap_still_rejected(self):
        from checks.test_map_suite import resign
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);manifest=suite.read(root/'inputs/validation.json')
            scene=copy.deepcopy(suite.read(root/'inputs/evaluation.json')['scenarios'][0]);scene.update(scenario_id='different_name',scenario_seed=912345)
            manifest['scenarios'][0]=scene;write_json(root/'inputs/validation.json',manifest);resign(root,plan)
            with self.assertRaisesRegex(ValueError,'overlap'):suite.load_plan(root)

    def test_changed_intermediate_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);synthetic_training(root,plan);job=next(j for j in plan['jobs'] if j['seed'] is not None)
            (root/job['train']/'checkpoint_episode_000001.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checkpoint inventory'):suite.training_receipt(root,plan,job)

    def test_full_six_group_report_with_diagnostics_and_safe_reentry(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);synthetic_training(root,plan)
            with patch.object(cv,'evaluate_point',side_effect=fake_point) as evaluate:
                cv.diagnose(root,plan,plan['jobs']);self.assertEqual(evaluate.call_count,16)
            with patch.object(cv,'evaluate_point',side_effect=AssertionError('must skip')):cv.diagnose(root,plan,plan['jobs'])
            for job in plan['jobs']:
                trained=None if job['seed'] is None else suite.training_receipt(root,plan,job)
                evaluation_rows(root,plan,job,suite.read(root/'inputs/evaluation.json'),root/job['evaluation'],'zero' if trained is None else trained['model_state_sha256'],3)
            report=summarize(root,root/'report')
            self.assertEqual(len(report['groups']),6);self.assertTrue((root/'report/validation_curves.csv').is_file())
            job=next(j for j in plan['jobs'] if j['seed'] is not None)
            self.assertIn('Fixed validation',(root/'jobs'/job['name']/'diagnostics/learning_curves.html').read_text())

    def test_missing_diagnostic_cannot_execute_during_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);synthetic_training(root,plan);job=next(j for j in plan['jobs'] if j['seed'] is not None)
            with patch.object(cv,'evaluate_point',side_effect=AssertionError('automatic restore forbidden')):
                with self.assertRaisesRegex(ValueError,'missing diagnostic'):cv.point_rows(root,plan,job,0,suite.training_receipt(root,plan,job),allow_execute=False)

    def test_diagnostic_summary_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder);synthetic_training(root,plan);job=next(j for j in plan['jobs'] if j['seed'] is not None)
            with patch.object(cv,'evaluate_point',side_effect=fake_point):cv.diagnose(root,plan,[job])
            path=root/'jobs'/job['name']/'diagnostics/convergence.json';value=suite.read(path);value['curve'][-1]['success_rate']=0.;write_json(path,value)
            with self.assertRaisesRegex(ValueError,'summary/source/curve'):cv.verified_diagnostic(root,plan,job)

    def test_mechanism_launcher_cannot_restore_diagnostic_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root,plan=fixture(folder)
            with self.assertRaisesRegex(ValueError,'not automatic mechanism'):suite.run(root,stage='diagnose')
