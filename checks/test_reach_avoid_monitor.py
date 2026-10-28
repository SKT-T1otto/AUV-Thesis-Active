"""Observer tests with synthetic packets/fake runtimes; ZERO native initializations."""
import ast
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from project_paths import ROOT
from chapter3_bser.reach_avoid.learning_diagnostics import settings,schedule,episode_metrics,aggregate,assess,read_journal,create_json
from chapter3_bser.reach_avoid.monitoring import TrainingMonitor,evaluate_development,training_digest
from chapter3_bser.reach_avoid.monitor_reports import render


def summary(reason='timeout',steps=1,trace=None):
    return dict(actual_length=steps,termination_reason=reason,success=reason=='success',found=False,
        team_discounted_return=1.,team_undiscounted_return=1.,first_collision_agent_ids=[3] if reason=='obstacle_collision' else [],
        first_collision_phase='Search' if reason=='obstacle_collision' else None,
        reach_avoid=dict(trace=[] if trace is None else trace))


def curve(success=1.,collision=0.,timeout=0.,stagnation=0.):
    return [dict(episode=i*50,updates=32+i,development_sha256='fixed',episodes=12,
        success_rate=success,collision_rate=collision,timeout_rate=timeout,stagnation_rate=stagnation,
        uncertified_fraction=0.,team_discounted_return=10.,penalized_completion_steps=100.) for i in range(6)]


class FakeRuntime:
    def __init__(self,*args,**kwargs):
        self.step=0;self.terminal=False;self.ra_records=[]
        random.random();np.random.random();torch.rand(1)
    def advance(self,policy,explore):
        if explore:raise AssertionError('development cannot explore')
        random.random();np.random.random();torch.rand(1)
        self.step+=1;self.terminal=True
    def summary(self):return summary('success')
    def close(self):pass


def learned():
    from checks.reach_avoid_fixtures import learning_fixture
    learner,options,_=learning_fixture()
    # Four updates maximum per temporary learner, exactly as authorized.
    for _ in range(4):learner.update()
    return learner,options


class ReachAvoidMonitoring(unittest.TestCase):
    def test_schedule_500_includes_pretrained_initial_and_final(self):
        self.assertEqual(schedule(500,50),list(range(0,501,50)))
        self.assertEqual(schedule(51,50),[0,50,51])
    def test_invalid_settings_rejected(self):
        for change in (dict(tail_points=2),dict(eval_every=0),dict(rate_tolerance=float('nan')),dict(extra=1)):
            with self.assertRaises(ValueError):settings(dict(settings(),**change))
    def test_flat_zero_success_is_ineffective(self):
        r=assess(curve(0.,0.,1.),settings(),400)
        self.assertEqual(r['status'],'stable_but_ineffective');self.assertFalse(r['convergence_candidate'])
    def test_flat_collision_is_ineffective(self):
        self.assertEqual(assess(curve(0.,1.,0.),settings(),400)['status'],'stable_but_ineffective')
    def test_flat_stagnation_not_effective(self):
        self.assertFalse(assess(curve(stagnation=1.),settings(),400)['convergence_candidate'])
    def test_changing_curve_not_plateau(self):
        c=curve();c[-1]['success_rate']=.5
        self.assertEqual(assess(c,settings(),400)['status'],'still_changing')
    def test_empirical_candidate_requires_task_quality(self):
        c=curve();c[0].update(success_rate=.5,team_discounted_return=0.,penalized_completion_steps=400.)
        r=assess(c,settings(),400)
        self.assertTrue(r['convergence_candidate']);self.assertTrue(r['online_task_gain'])
        self.assertFalse(r['theoretical_convergence_proved'])
    def test_no_update_not_learning(self):
        c=curve()
        for r in c:r['updates']=32
        self.assertEqual(assess(c,settings(),400)['status'],'no_online_learning_updates')
    def test_safety_regression_reported(self):
        c=curve(success=.9,collision=.09);c[0]['collision_rate']=0.
        self.assertEqual(assess(c,settings(),400)['status'],'stable_with_safety_regression')
    def test_changed_development_population_rejected(self):
        c=curve();c[-1]['development_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'development set changed'):assess(c,settings(),400)
    def test_no_early_convergence_claim(self):
        self.assertEqual(assess(curve()[:3],settings(),400)['status'],'insufficient_evaluations')
    def test_failed_episodes_not_dropped(self):
        rows=[episode_metrics(summary('obstacle_collision'),400,settings())]
        with self.assertRaisesRegex(ValueError,'incomplete'):aggregate(rows,12)
        self.assertEqual(aggregate(rows,1)['executor_search_collision_count'],1)
    def test_collision_cannot_be_hidden_as_timeout(self):
        row=summary();row['first_collision_agent_ids']=[3]
        with self.assertRaises(ValueError):episode_metrics(row,400,settings())
    def test_time_penalty_does_not_reward_early_crash(self):
        self.assertEqual(episode_metrics(summary('obstacle_collision'),400,settings())['penalized_completion_steps'],400)
    def test_stall_uses_active_roles_and_goal_progress(self):
        goal=[5.,0.,0.,0.,0.,0.,1.,0.,0.,0.,5.,0.,0.]
        trace=[dict(agent_id=3,phase=1,action_mode='NOMINAL',selected_goal=goal,public_position=[0.,0.,0.],velocity=[0.,0.,0.]) for _ in range(50)]
        self.assertEqual(episode_metrics(summary(steps=50,trace=trace),400,settings())['stagnation'],1)
        for x in trace:x['local_goal_reached']=True
        self.assertEqual(episode_metrics(summary(steps=50,trace=trace),400,settings())['stagnation'],0)
        for x in trace:x['phase']=3;x['local_goal_reached']=False
        self.assertEqual(episode_metrics(summary(steps=50,trace=trace),400,settings())['active_agent_steps'],0)
    def test_wilson_uncertainty_visible_at_perfect_small_sample(self):
        r=aggregate([episode_metrics(summary('success'),400,settings())]*12,12)
        self.assertLess(r['success_ci95'][0],.8);self.assertAlmostEqual(r['success_ci95'][1],1.)
    def test_journal_ignores_only_unfinished_tail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log';p.write_bytes(b'{"episode":1}\n{"episode":')
            rows,receipt=read_journal(p)
            self.assertEqual(rows,[dict(episode=1)]);self.assertTrue(receipt['unfinished_tail'])
            p.write_bytes(b'not json\n')
            with self.assertRaises(json.JSONDecodeError):read_journal(p)
    def test_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.json';create_json(p,{})
            with self.assertRaises(FileExistsError):create_json(p,{})
    def test_development_evaluation_preserves_rng_training_and_next_update(self):
        from checks.reach_avoid_fixtures import learning_fixture
        a,o,_=learning_fixture();b,_,_=learning_fixture()
        a.options['minimum_updates']=3;b.options['minimum_updates']=3;o['minimum_updates']=3
        for _ in range(3):a.update();b.update()
        before=training_digest(a);random.seed(27);np.random.seed(27);torch.manual_seed(27)
        py=random.getstate();npstate=copy.deepcopy(np.random.get_state());ts=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as d,patch('chapter3_bser.reach_avoid.runtime.ReachAvoidRuntime',FakeRuntime),patch('core.env.uav_env.UAVEnv.__init__',side_effect=AssertionError('native forbidden')):
            r=evaluate_development(a,a.state(),dict(max_steps=400),[dict(scenario_id='dev',scenario_seed=2)],o,settings(),Path(d)/'eval')
        self.assertEqual(r['success_rate'],1.);self.assertEqual(training_digest(a),before)
        self.assertEqual(random.getstate(),py);np.testing.assert_equal(np.random.get_state(),npstate);self.assertTrue(torch.equal(ts,torch.get_rng_state()))
        a.update();b.update();self.assertEqual(training_digest(a),training_digest(b))
    def test_monitor_snapshots_live_final_reports_and_tamper(self):
        learner,o=learned()
        with tempfile.TemporaryDirectory() as d,patch('chapter3_bser.reach_avoid.runtime.ReachAvoidRuntime',FakeRuntime),patch('chapter3_bser.reach_avoid.monitor_reports.plots',return_value=dict(available=False)):
            root=Path(d);source=dict(review_sha256='synthetic',profile='fixture')
            create_json(root/'run.json',dict(source_identity=source))
            m=TrainingMonitor(root,source,dict(max_steps=400),[dict(scenario_id='dev',scenario_seed=2)],o,
                dict(settings(),eval_every=1),1,'devsha',{'train'},{'fingerprint'})
            try:
                m.start(learner);m.after_pretrain(learner);m.before_episode(learner)
                m.on_episode(learner,summary(),1,1,['train'],['fingerprint'])
                live=render(root,root/'live');self.assertFalse(live['training_completed']);self.assertFalse(live['assessment']['convergence_candidate'])
                create_json(root/'completed.json',dict(source_identity=source,completed=True,episodes=1))
                m.finish(learner,['train'],['fingerprint'])
                final=json.loads((root/'monitoring/final_report/report.json').read_text())
                self.assertTrue(final['training_completed']);self.assertEqual(final['evaluation_points'],[0,1])
                with self.assertRaises(FileExistsError):render(root,root/'live')
                path=root/'monitoring/evaluations/episode_000001/episodes.jsonl';path.write_text('tampered\n')
                with self.assertRaisesRegex(ValueError,'raw development'):render(root,root/'tampered_report')
            finally:m.close()
    def test_future_training_overlap_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'dev.json';create_json(p,dict(scenarios=[dict(scenario_id='same')]))
            args=NS(dev_scenarios=p,monitor_settings=ROOT/'configs/reach_avoid_monitor.json',config='unused',episodes=500)
            with patch('chapter3_bser.reach_avoid.monitoring.inputs',return_value=({},[dict(scenario_id='same')])),patch('chapter3_bser.reach_avoid.monitoring.scenario_fingerprint',return_value='same'):
                with self.assertRaisesRegex(ValueError,'overlaps'):TrainingMonitor.from_args(args,root/'out',{}, {},[dict(scenario_id='same')],{},[],[])
            self.assertFalse((root/'out').exists())
    def test_observer_guard_does_not_remove_hidden_update(self):
        from tools.reach_avoid_monitor_source import workflow_without_observer
        text=(ROOT/'chapter3_bser/reach_avoid/workflow.py').read_text()
        original=ast.dump(workflow_without_observer(text),include_attributes=False)
        altered=text.replace('if monitor:monitor.start(learner)','if monitor:\n        monitor.start(learner)\n        learner.update()')
        self.assertNotEqual(original,ast.dump(workflow_without_observer(altered),include_attributes=False))
    def test_experiment_dry_run_has_only_latest_500ep(self):
        from tools.reach_avoid_experiment import main
        with patch('chapter3_bser.reach_avoid.workflow.admission',side_effect=AssertionError('must not execute')):
            result=main(['--validation','not_loaded','--output','not_created'])
        self.assertFalse(result['executed']);self.assertEqual(result['online_training_episodes'],500)
        self.assertEqual(result['evaluation_points'],list(range(0,501,50)))
        self.assertFalse(result['other_algorithm_experiments'])
