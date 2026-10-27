"""Safety contracts and failure mechanisms; no environment initializations."""
import copy,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from core.env.public_events import PublicEventJournal,ScanObservation
from chapter3_bser.safe_rl.options import DEFAULTS,validate,FRAME_DIM
from chapter3_bser.safe_rl.dynamics import prior,nominal,integrate,brake,stopping_bound,speed_limit
from chapter3_bser.safe_rl.evidence import ScanEvidence,EvidenceGeometry
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.learner import SafetyLearner,horizon_labels
from tools.audit_safe_speed import declared_specs,run as audit
from project_paths import ROOT

SPECS=declared_specs()
CONTRACT=dict(dt=.2,eps=1e-8,bounds=(20.,20.,8.),obstacle_sensor_range=4.5,
    prior_kv_xy=1.1,prior_kv_z=1.,prior_slow_radius_xy=2.4,prior_slow_radius_z=1.2,
    prior_strength_search=1.,prior_strength_executor=1.,residual_scale_search=.2,residual_scale_executor=.15,use_residual_prior=True)

def state(occupied=(),unknown=()):
    cells=np.indices((10,10,8)).reshape(3,-1).T
    centers=(cells+.5)*np.array([2.,2.,1.]);mask=np.zeros(800,bool);mask[list(occupied)]=True
    unseen=np.zeros(800,bool);unseen[list(unknown)]=True;free=~(mask|unseen)
    return NS(step=0,target_found=False,map_revision=1,
        grid=NS(shape=(10,10,8),origin=(0.,0.,0.),spacing=(2.,2.,1.),cell_centers=centers),
        occupancy=NS(occupied_mask=mask,free_mask=free,unknown_mask=unseen),
        agents=tuple(NS(agent_id=i,position=(3.+i,3.,4.),velocity=(0.,0.,0.)) for i in range(4)))

def guidance():
    return NS(agent_assignments=tuple(NS(agent_id=i,tracking_waypoint=(9.,3.,4.),final_waypoint=(9.,3.,4.),
        hold_position=(3.+i,3.,4.),hold_state=False,reachable=True,execution_request=False) for i in range(4)))

class SafeDynamics(unittest.TestCase):
    def test_speed_audit_flags_fast_two_without_changing_physical_limits(self):
        a=audit();self.assertEqual(a['environment_initializations'],0)
        self.assertEqual([r['independent_axis_bounds_fit_range'] for r in a['rows']],[False,False,True,True])
        self.assertAlmostEqual(a['rows'][0]['physical_v_xy_max'],2.8)
        self.assertAlmostEqual(a['rows'][0]['horizontal_speed_cap_at_full_range'],2.4699024,places=6)
    def test_allowable_speed_shrinks_with_evidence_distance(self):
        for spec in SPECS:
            caps=[speed_limit(d,spec,.2,.18) for d in (0.,.1,.5,2.,4.)]
            self.assertEqual(caps[0],0.);self.assertTrue(all(a<=b for a,b in zip(caps,caps[1:])))
    def test_latency_increases_braking_budget(self):
        self.assertGreater(stopping_bound(1.,SPECS[3],.2,.18,reaction_steps=20),stopping_bound(1.,SPECS[3],.2,.18,reaction_steps=1))
    def test_full_authority_can_reverse_a_dangerous_prior(self):
        s=SPECS[3];u,_,_=nominal([0,0,4],[0,0,0],[20,0,4],[-1,0,0],s,CONTRACT,3)
        self.assertAlmostEqual(u[0],.68)
        self.assertLess(brake([1,0,0],s,.2)[0],0.)
    def test_integrator_uses_post_acceleration_drag(self):
        p,v=integrate([1,2,3],[.5,.2,-.1],[.2,-.1,.3],SPECS[3],.2,[.18,-.18])
        expected=np.array([.54,.18,-.04]);expected[:2]+=-.16*expected[:2]*.2+np.array([.18,-.18])*.2
        expected[2]+=-.24*expected[2]*.2+.02*.2
        np.testing.assert_allclose(v,expected,atol=1e-12);np.testing.assert_allclose(p,np.array([1,2,3])+.2*expected)
    def test_brake_accounts_for_known_buoyancy(self):
        _,v=integrate([0,0,4],[.02,0,.02],brake([.02,0,.02],SPECS[3],.2),SPECS[3],.2)
        np.testing.assert_allclose(v,0.,atol=1e-12)
    def test_conservative_budget_covers_adverse_discrete_motion(self):
        for spec in SPECS:
            p=np.zeros(3);v=np.array([spec['v_xy_max'],0.,0.]);start=v[0]
            for _ in range(200):
                p,v=integrate(p,v,brake(v,spec,.2),spec,.2,[np.sqrt(2)*.18,0.])
                if v[0]<=.052:break
            self.assertLessEqual(p[0],stopping_bound(start,spec,.2,.18))
    def test_no_braking_authority_is_infeasible(self):
        s=dict(SPECS[3],a_xy_max=.1)
        self.assertEqual(stopping_bound(1.,s,.2,.18),float('inf'))
        self.assertLess(speed_limit(1.,s,.2,.18),1e-8)
    def test_combined_motion_is_not_claimed_certified_by_axis_audit(self):
        self.assertIn('independent axes',audit()['scope'])

class SafeEvidence(unittest.TestCase):
    def event(self,hit=True):
        journal=PublicEventJournal('unit');journal.reset()
        journal.append(0,'existing_sensor','scan',ScanObservation(((3.,3.,4.),),((1.,0.,0.),),((1.,),),((hit,),),(0,)))
        return journal.read()
    def test_public_hit_survives_free_grid_label(self):
        e=ScanEvidence();e.consume(self.event());g=EvidenceGeometry(state(),e,validate())
        self.assertTrue(g.local([4,3,4])[1]);self.assertEqual(g.distance([4,3,4]),0.)
        self.assertFalse(g.segment([3,3,4],[4,3,4]))
    def test_no_return_is_not_recorded_as_obstacle(self):
        e=ScanEvidence();e.consume(self.event(False));self.assertEqual(len(e.points()),0)
    def test_journal_replay_is_rejected(self):
        e=ScanEvidence();events=self.event();e.consume(events)
        with self.assertRaisesRegex(ValueError,'cursor'):e.consume(events)
    def test_retained_hit_memory_is_bounded_and_evictions_visible(self):
        e=ScanEvidence(capacity=1);events=self.event();e.consume(events)
        journal=PublicEventJournal('unit');journal.reset();journal.append(0,'x','scan',events[0].payload)
        journal.append(1,'x','scan',ScanObservation(((3.,3.,4.),),((1.,0.,0.),),((2.,),),((True,),),(0,)))
        e.consume(journal.read(1));self.assertEqual(len(e.hits),1);self.assertEqual(e.evictions,1)
    def test_unknown_grid_remains_uncertain(self):
        g=EvidenceGeometry(state(unknown=range(800)),ScanEvidence(),validate())
        unknown,free=g.local([3,3,4]);self.assertEqual(unknown,1.);self.assertFalse(free)
    def test_closed_contact_with_public_hit_is_blocked(self):
        e=ScanEvidence();e.hits[(0,0,0)]=(np.array([4.,3.,4.]),0)
        g=EvidenceGeometry(state(),e,validate())
        self.assertFalse(g.segment([4-g.padding,3,4],[4-g.padding,3,4]))

class SafeControl(unittest.TestCase):
    def test_high_risk_nominal_cannot_be_reaccepted_under_governor_name(self):
        learner=NS(ready=True,risk=lambda z,u:dict(score=.9),action=lambda z:np.zeros(3))
        c=SafetyController(SPECS,CONTRACT,validate(),learner=learner)
        _,_,rows,_=c.act(state(),guidance(),np.zeros((4,3)),())
        self.assertTrue(all(r['mode']=='UNCERTIFIED_BRAKE' for r in rows))
        self.assertTrue(all(not r['model_checked'] for r in rows))
    def test_all_roles_get_independent_safety_features(self):
        c=SafetyController(SPECS,CONTRACT,validate());u,z,rows,g=c.act(state(),guidance(),np.zeros((4,3)),())
        self.assertEqual(z.shape,(4,192));self.assertEqual(u.shape,(4,3));self.assertEqual(len(rows),4)
        self.assertEqual(c.counts['agent_checks'],4);self.assertTrue(np.isfinite(z).all())
        self.assertTrue(all(r['safety_guarantee'] is False for r in rows))
    def test_obstacle_in_goal_prevents_nominal_motion_for_every_role(self):
        c=SafetyController(SPECS,CONTRACT,validate())
        for i,a in enumerate(state().agents):c.evidence.hits[(i,0,0)]=(np.array(a.position)+[.7,0,0],0)
        guide=guidance()
        for i,item in enumerate(guide.agent_assignments):item.final_waypoint=tuple(np.asarray(state().agents[i].position)+[.7,0,0])
        u,z,rows,g=c.act(state(),guide,np.zeros((4,3)),())
        self.assertTrue(all(r['mode']!='NOMINAL' for r in rows))
        self.assertTrue(all(np.all(np.abs(u[i])<=1.) for i in range(4)))
    def test_safety_control_is_not_disabled_by_task_zero_action(self):
        st=state();st.agents=tuple(NS(agent_id=i,position=(3.+i,3.,4.),velocity=(.8,0.,0.)) for i in range(4))
        c=SafetyController(SPECS,CONTRACT,validate())
        for i,a in enumerate(st.agents):c.evidence.hits[(i,0,0)]=(np.array(a.position)+[.55,0,0],0)
        u,z,rows,g=c.act(st,guidance(),np.zeros((4,3)),())
        self.assertTrue(all(u[i,0]<0 for i in range(4)))
    def test_hold_anchor_does_not_require_reaching_semantic_goal(self):
        c=SafetyController(SPECS,CONTRACT,validate());c.anchors[3]=np.array([5,3,4.])
        _,_,rows,_=c.act(state(),guidance(),np.zeros((4,3)),())
        self.assertEqual(rows[3]['semantic_goal'],[9.,3.,4.]);np.testing.assert_array_equal(c.anchors[3],[6,3,4.])
    def test_backup_horizon_failure_is_not_marked_safe(self):
        c=SafetyController(SPECS,CONTRACT,validate())
        with patch('chapter3_bser.safe_rl.control.brake',return_value=np.array([.8,0,0])):
            ok,result=c.check([3,3,4],[.8,0,0],[.8,0,0],SPECS[3],EvidenceGeometry(state(),ScanEvidence(),validate()),[0,0])
        self.assertFalse(ok);self.assertNotEqual(result['reason'],'model_checked_stop')
    def test_original_nearest_truth_distance_is_not_a_safety_feature(self):
        self.assertEqual(FRAME_DIM,48)
        for name in ('control.py','evidence.py','dynamics.py','navigation.py'):
            text=(ROOT/'chapter3_bser/safe_rl'/name).read_text()
            self.assertNotIn('_nearest_obstacle_distance(',text);self.assertNotIn('ground_truth_obstacles',text)

class SafeLearning(unittest.TestCase):
    def test_other_agent_failure_is_censored_not_own_positive(self):
        costs=np.zeros((3,4));costs[-1,1]=1
        y,m=horizon_labels(costs,8)
        self.assertEqual(y[0,3].sum(),0);self.assertEqual(m[0,3].sum(),3)
        np.testing.assert_array_equal(y[0,1],[0,0,1,1,1,1,1,1])
    def test_timeout_does_not_invent_unobserved_safe_horizons(self):
        y,m=horizon_labels(np.zeros((2,4)),8)
        self.assertEqual(m[1,0].sum(),1);self.assertEqual(y.sum(),0)
    def test_own_collision_labels_longer_horizons_positive(self):
        c=np.zeros((2,4));c[0,3]=1;y,m=horizon_labels(c,8)
        self.assertEqual(y[0,3].sum(),8);self.assertEqual(m[0,3].sum(),8)
    def test_safety_initialization_preserves_task_torch_rng(self):
        before=torch.random.get_rng_state().clone();SafetyLearner(validate())
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
    def test_four_complete_updates_use_actual_actions_and_are_finite(self):
        o=dict(validate(),batch_size=4,warmup_samples=4,minimum_updates=4);model=SafetyLearner(o)
        rows=[]
        for t in range(6):rows.append(dict(safety_features=np.full((4,192),t/10,np.float32),
            normalized_executed_acceleration=np.full((4,3),-.2,np.float32),collision_cost=np.array([0,0,0,t==5]),
            safety_rows=[dict(mode='NOMINAL' if t==4 else 'BRAKE',model_checked=True) for _ in range(4)]))
        model.add_episode(rows);before=[p.detach().clone() for p in model.actor.parameters()]
        for _ in range(4):
            metrics=model.update();self.assertTrue(np.isfinite(metrics['risk_loss']));self.assertTrue(np.isfinite(metrics['actor_loss']))
        self.assertTrue(model.ready);self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.actor.parameters())))
        self.assertEqual(model.replay[0][1].tolist(),np.full(3,-.2,np.float32).tolist())
    def test_untrained_policy_cannot_be_loaded_as_learned(self):
        model=SafetyLearner(validate())
        with self.assertRaisesRegex(ValueError,'untrained'):model.load_for_evaluation(model.state())
    def test_invalid_physical_assumption_rejected(self):
        with self.assertRaisesRegex(ValueError,'flow'):validate(dict(DEFAULTS,flow_component_bound=.1))
    def test_old_actor_dimensions_are_not_relabelled(self):
        from chapter3_bser.learning.config import validate as parent_validate
        c=json.loads((ROOT/'configs/b2_mechanism.json').read_text());self.assertEqual(parent_validate(c)['observation_dim'],28)
        self.assertEqual(SafetyLearner(validate()).input_dim,192)

class SafeEntryPoints(unittest.TestCase):
    def test_new_absolute_input_factory_preserves_original_composition_metadata(self):
        from chapter3_bser.safe_rl.runtime import SafetyRuntime
        runtime=SafetyRuntime.__new__(SafetyRuntime);runtime._safety_episode_id=2
        c=json.loads((ROOT/'configs/b2_mechanism.json').read_text())
        with patch('chapter3_bser.safe_rl.runtime.MissionCoreEnv') as factory:
            runtime._make_environment(c)
            kw=factory.call_args.kwargs
            self.assertIs(kw['use_residual_prior'],False);self.assertEqual(kw['residual_scale_executor'],1.)
            self.assertEqual(runtime._original_composition['residual_scale_executor'],.15)
            factory.return_value.enable_public_event_capture.assert_called_once()
    def test_train_and_evaluate_are_dry_without_execute(self):
        from chapter3_bser.safe_rl.workflow import main
        with patch('chapter3_bser.safe_rl.workflow.execute',side_effect=AssertionError('must not execute')):
            for command in ('train','evaluate','collect'):
                result=main([command,'--config','missing','--scenarios','missing','--output','missing','--validation','missing','--episodes','1'])
                self.assertFalse(result['executed'])
    def test_training_refuses_development_scenarios(self):
        from chapter3_bser.safe_rl.workflow import inputs
        config=json.loads((ROOT/'configs/b2_mechanism.json').read_text())
        manifest=dict(scenarios=[dict(scenario_id='eval_1',scenario_role='eval',scenario_split='eval')])
        with patch('chapter3_bser.safe_rl.workflow.read',side_effect=[config,manifest]):
            with self.assertRaisesRegex(ValueError,'train/train'):inputs('config','manifest',training=True,episodes=1)
    def test_linux_validation_cannot_be_replaced_by_windows_evidence(self):
        from chapter3_bser.safe_rl.workflow import admission
        with patch('chapter3_bser.safe_rl.workflow.verify_reviewed_changes',return_value={}),patch('chapter3_bser.safe_rl.workflow.research_readiness'),patch('chapter3_bser.learning.map_suite.platform.system',return_value='Windows'):
            with self.assertRaisesRegex(ValueError,'Linux'):admission('missing_validation.json')
