"""Finite analytic oracles; no simulator episodes, training or checkpoints."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from core.env.public_events import PublicEvent, StepObservation, ScanObservation, payload_digest, vectors
from core.env.target_motion import swept_relative_min_distance, TargetState, advance_target_state
from chapter3_bser.types import SearchCandidate
from chapter3_bser.experiments.coupled_search_v1.core_config import CORE_DEFAULT, validate_core_config
from chapter3_bser.experiments.coupled_search_v1.coupled_belief import CoupledBelief
from chapter3_bser.experiments.coupled_search_v1.geometry_hypotheses import GeometryHypothesis, compatible
from chapter3_bser.experiments.coupled_search_v1.execution_forecast import nominal_step, PublicExecutionErrors, forecast_route
from chapter3_bser.experiments.coupled_search_v1.discovery import signature, signature_distance, compress, team_score, select_team
from chapter3_bser.experiments.coupled_search_v1.types import PublicStepEvidence, TargetForecast, ExecutionForecast, RouteProposal, DiscoverySignature
from chapter3_bser.experiments.coupled_search_v1.numerics import Budget, BudgetExceeded, swept_distances, locked
from chapter3_bser.experiments.coupled_search_v1.session import CoupledSearchSession
from chapter3_bser.experiments.coupled_search_v1.region_proposals import remaining_cost
from checks.test_public_events import guidance


def contract():
    return dict(bounds=(20.,20.,10.),dt=.2,eps=1e-6,prior_kv_xy=1.2,prior_kv_z=1.4,
        prior_slow_radius_xy=2.,prior_slow_radius_z=1.,prior_strength_search=1.,use_residual_prior=True,
        detect_eps_bias=0.,target_motion_mode='constant_velocity_reflect_v1',target_obstacle_clearance=0.,
        target_max_reflections_per_step=8,obstacle_sensor_range=4.)


def specs():
    return tuple(dict(a_xy_max=.4+i*.1,a_z_max=.3,v_xy_max=1.+i*.1,v_z_max=.5,
        drag_xy=.1,drag_z=.2,buoyancy_bias=.02,sensor_range=1.) for i in range(4))


def evidence(step=1,**changes):
    p=vectors([[2.,1.,1.],[18.,18.,9.],[18.,18.,9.],[18.,18.,9.]])
    obs=StepObservation(p,p,vectors(np.zeros((4,3))),vectors(np.zeros((4,3))),False,False,True,True,False,'running')
    obs=replace(obs,**changes)
    event=PublicEvent('e',step,step,'detector','step',obs,payload_digest(obs))
    return PublicStepEvidence('e',step,(),event)


def prior():
    b=CoupledBelief(copy.deepcopy(CORE_DEFAULT),contract(),specs())
    b.initialize_hypotheses((GeometryHypothesis(()),)*2,
        [[[2.,1.,1.],[10.,1.,1.]],[[12.,1.,1.],[13.,1.,1.]]],np.zeros((2,2,3)),[.5,.5],[[.5,.5],[.5,.5]])
    return b


def sig(values,first=None,**changes):
    a=np.asarray(values,float)
    return DiscoverySignature(locked(a),locked(a>0 if first is None else first,dtype=int),
        changes.get('target','t'),changes.get('execution','e'))


class CoupledSearch(unittest.TestCase):
    def test_negative_evidence_changes_map_and_conditional_target_weights(self):
        b=prior();b.consume(evidence())
        np.testing.assert_allclose(b.alpha,[1/3,2/3]);np.testing.assert_equal(b.omega,[[0,1],[.5,.5]])

    def test_duplicate_transition_does_not_square_likelihood(self):
        b=prior();e=evidence();b.consume(e);identity=b.identity;b.consume(e)
        self.assertEqual(b.identity,identity);self.assertEqual(b.counts['joint_updates'],1)

    def test_collision_and_found_do_not_apply_negative_likelihood(self):
        for changes in (dict(target_transition_completed=False,discovery_check_completed=False,episode_terminated=True),
                        dict(found_after=True)):
            b=prior();b.consume(evidence(**changes));np.testing.assert_equal(b.alpha,[.5,.5]);self.assertTrue(b.stopped)

    def test_timeout_still_uses_completed_negative_check(self):
        b=prior();b.consume(evidence(episode_terminated=True,terminal_reason='timeout'))
        np.testing.assert_allclose(b.alpha,[1/3,2/3]);self.assertTrue(b.stopped)

    def test_total_degeneracy_is_visible_without_likelihood_floor(self):
        b=prior();b.positions[:]=[2.,1.,1.];b.config['replenish_attempts']=0;b.consume(evidence())
        self.assertFalse(b.valid);self.assertIn('zero_mass',b.error)
        with self.assertRaises(ValueError):b.forecast(Budget(5))

    def test_replenishment_requires_full_replay(self):
        b=prior();b.history_complete=False
        with self.assertRaisesRegex(ValueError,'history_truncated'):b._rebuild(Budget(5))

    def test_map_observation_and_negative_evidence_both_enter_map_weights(self):
        b=prior()
        dummy=SimpleNamespace(event_id=('scan',),payload='new scan')
        item=replace(evidence(),scans=(dummy,))
        with patch('chapter3_bser.experiments.coupled_search_v1.coupled_belief.compatible',side_effect=[False,True]):
            b.consume(item)
        np.testing.assert_equal(b.alpha,[0,1])

    def test_forecast_preserves_map_conditioning_and_original_reflections(self):
        b=prior();b.config['horizon']=2;b.positions[1,0]=[19.9,1,1];b.velocities[1,0]=[1,0,0]
        f=b.forecast(Budget(5));self.assertEqual(f.positions.shape,(4,3,3))
        s=advance_target_state(TargetState([19.9,1,1],[1,0,0],0,contract()['target_motion_mode']),.2,contract()['bounds'],(),max_reflections=8)
        np.testing.assert_allclose(f.positions[2,1],s.position);self.assertAlmostEqual(sum(f.weights),1)
        with self.assertRaises(ValueError):f.positions.setflags(write=True)

    def test_batch_swept_criterion_matches_independent_scalar_kernel(self):
        rng=np.random.default_rng(17);points=rng.normal(size=(4,30,3))
        d=swept_distances(*points)
        expected=[swept_relative_min_distance(*(x[i] for x in points))[0] for i in range(30)]
        np.testing.assert_allclose(d,expected,atol=1e-14)
        self.assertEqual(float(swept_distances([0,0,0],[0,0,0],[1,0,0],[1,0,0])),1.)

    def test_geometry_checks_hits_misses_and_world_boundaries(self):
        scan=ScanObservation(((1.,1.,1.),),((1.,0.,0.),),((2.,),),((True,),),(0,))
        box=GeometryHypothesis((((4.,1.,1.),(2.,1.,1.)),))
        self.assertTrue(compatible(box,scan,(10,10,10),4,.01))
        self.assertFalse(compatible(GeometryHypothesis(()),scan,(10,10,10),4,.01))
        self.assertFalse(compatible(box,replace(scan,hit_mask=((False,),),distances=((4.,),)),(10,10,10),4,.01))

    def test_heterogeneous_forecast_uses_acceleration_drag_and_caps(self):
        p,v=nominal_step([1,1,1],[0,0,0],[10,1,1],specs()[0],contract())
        self.assertAlmostEqual(v[0],.4*.2*(1-.1*.2));self.assertAlmostEqual(p[0],1+v[0]*.2)
        _,faster=nominal_step([1,1,1],[0,0,0],[10,1,1],specs()[2],contract())
        self.assertGreater(faster[0],v[0])

    def test_public_error_scenarios_keep_temporal_and_cross_agent_blocks(self):
        cfg=copy.deepcopy(CORE_DEFAULT);cfg['horizon']=4;errors=PublicExecutionErrors(cfg,specs(),contract())
        errors.history.extend([np.full((3,3),.1),np.full((3,3),.2)])
        bank=errors.scenarios(3)
        np.testing.assert_allclose(bank.errors[1,:,0,0],[.1,.2,.1,.2])
        np.testing.assert_equal(bank.errors[1,:,0],bank.errors[1,:,2]);self.assertFalse(np.any(bank.errors[0]))

    def test_unsafe_forecast_and_hold_have_no_future_discovery_credit(self):
        cfg=copy.deepcopy(CORE_DEFAULT);cfg['horizon']=2;bank=PublicExecutionErrors(cfg,specs(),contract()).scenarios(0)
        c=SearchCandidate(0,'x',(10,1,1),locked([[1,1,1],[10,1,1]]),locked([],dtype=int),9,9,9,'test')
        a=SimpleNamespace(position=(1,1,1),velocity=(0,0,0),agent_id=0)
        for r in (RouteProposal(c),RouteProposal(c,blocked=True)):
            f=forecast_route(r,a,specs()[0],contract(),bank,SimpleNamespace(segment_free=lambda a,b:False),Budget(5))
            self.assertFalse(np.any(f.valid_steps))

    def test_first_hit_discount_and_team_union_never_double_count(self):
        t=TargetForecast(locked([[[0,0,0],[0,0,0],[0,0,0]]]),locked([1]),(0,),0,'t')
        e=ExecutionForecast(locked([[[3,0,0],[.5,0,0],[0,0,0]]]),locked([2],dtype=int),('horizon',),'e')
        s=signature(t,e,1,dt=.2,beta=2)
        self.assertEqual(s.first_steps[0,0],1);self.assertAlmostEqual(s.values[0,0],np.exp(-.4))
        score=team_score([s,s],[1],[1],[0,0],rho=.5,lambda_cost=0)
        self.assertAlmostEqual(score.objective,np.exp(-.4));self.assertEqual(score.detection_probability,1)

    def test_risk_and_cost_objective_matches_hand_computation(self):
        s=sig([[1,0],[0,.5]])
        score=team_score([s],[.25,.75],[.5,.5],[2],rho=.4,lambda_cost=.1)
        self.assertAlmostEqual(score.objective,.6*(.25+.375)/2+.4*.25-.2)
        self.assertAlmostEqual(score.detection_probability,.5)

    def test_shared_sample_identity_is_enforced(self):
        with self.assertRaisesRegex(ValueError,'share'):
            signature_distance(sig([[1]]),sig([[1]],execution='different'),[1],0,0,0)

    def test_compression_retains_reference_and_reports_actual_unmet_epsilon(self):
        ss=[sig([[x]]) for x in (0,.2,.7,1)]
        result=compress(ss,[1],[0]*4,limit=2,epsilon=.01,lambda_cost=0,reference_indices=(0,))
        self.assertEqual(result.indices,(0,3));self.assertAlmostEqual(result.epsilon,.3)
        self.assertFalse(result.reached_error_target)

    def test_compression_finite_pool_objective_loss_is_bounded(self):
        ss=[sig([[x]]) for x in (0,.2,.7,1)]
        r=compress(ss,[1],[0]*4,limit=2,epsilon=.01,lambda_cost=0,reference_indices=(0,))
        selected,score,count=select_team([r.indices,(0,)],ss,[1],[1],[0]*4,rho=.25,lambda_cost=0,budget=Budget(5))
        self.assertEqual(count,2);self.assertLessEqual(1-score.objective,2*r.epsilon+1e-12)

    def test_joint_selection_observes_frozen_group_and_cost(self):
        ss=[sig([[1],[0]]),sig([[0],[1]]),sig([[1],[0]])]
        chosen,score,count=select_team([(0,1),(2,)],ss,[.5,.5],[1],[0,0,0],rho=0,lambda_cost=0,budget=Budget(5))
        self.assertEqual(chosen,(1,2));self.assertEqual(score.detection_probability,1);self.assertEqual(count,2)

    def test_installed_path_replacement_invalidates_prediction(self):
        s=CoupledSearchSession(copy.deepcopy(CORE_DEFAULT),specs(),contract());context=guidance()
        s.prediction=dict(step=0,paths={0:tuple(context.agent_assignments[0].planned_path)},
                          finals={0:context.agent_assignments[0].final_waypoint},forecasts={},installed_hash=None)
        s.installed(context,{0:1});self.assertIsNotNone(s.prediction)
        s.installed(guidance(path=((0,0,0),(.5,1,0),(1,0,0))))
        self.assertIsNone(s.prediction);self.assertEqual(s.counts['forecast_invalidations'],1)

    def test_first_install_checks_final_waypoint_as_well_as_full_path(self):
        s=CoupledSearchSession(copy.deepcopy(CORE_DEFAULT),specs(),contract());context=guidance()
        s.prediction=dict(step=0,paths={0:tuple(context.agent_assignments[0].planned_path)},
                          finals={0:(2.,0.,0.)},forecasts={},installed_hash=None)
        s.installed(context)
        self.assertIsNone(s.prediction)

    def test_budget_and_core_configuration_are_explicit(self):
        validate_core_config(copy.deepcopy(CORE_DEFAULT));b=Budget(5,queries=1);b.query()
        with self.assertRaises(BudgetExceeded):b.query()
        c=copy.deepcopy(CORE_DEFAULT);c['rho']=1.01
        with self.assertRaises(ValueError):validate_core_config(c)

    def test_installed_segment_cost_does_not_require_cheapest_graph_connector(self):
        agent=SimpleNamespace(agent_id=0,role='searcher',position=(.4,.4,.4))
        graph=SimpleNamespace(cell_centers=np.asarray([[1,1,1]]),component_labels=np.asarray([0]),valid_mask=np.asarray([True]),
            endpoint_connectors=(),time_rates_for_role=lambda role:(2.,3.))
        state=SimpleNamespace(agents=(agent,),planning_graph=graph)
        c=SearchCandidate(0,'installed',(1.,1.,1.),locked([[0,0,0],[1,1,1]]),locked([0],dtype=int),2,99,99,'test')
        r=remaining_cost(state,RouteProposal(c,tracker_index=1,reference=True))
        expected=np.sqrt(.6**2+.6**2)*2+.6*3
        self.assertAlmostEqual(r.candidate.planning_cost,expected)
        np.testing.assert_equal(r.candidate.path_points,c.path_points)

    def test_replay_restarts_targets_before_reapplying_negative_history(self):
        b=prior();b.history.append(PublicStepEvidence('e',0,(),None));b.history.append(evidence())
        initial=b.positions.copy();velocity=b.velocities.copy();maps=b.maps
        def reset_prior(selected,budget):b.initialize_hypotheses(selected,initial,velocity,[.5,.5],[[.5,.5],[.5,.5]])
        with patch.object(b.geometry,'propose',return_value=maps),patch.object(b,'_prior',side_effect=reset_prior):
            b._rebuild(Budget(5));np.testing.assert_allclose(b.alpha,[1/3,2/3])
            b._rebuild(Budget(5));np.testing.assert_allclose(b.alpha,[1/3,2/3])

    def test_forecast_prior_matches_existing_torch_control_kernel(self):
        import torch
        from core.env.uav_env import _BaseUAVEnv
        c=contract();ss=specs()[:3]
        p=torch.tensor([[1.,2.,3.]]*3,dtype=torch.float64);v=torch.tensor([[.3,-.2,.1]]*3,dtype=torch.float64)
        targets=torch.tensor([[8.,3.,4.]]*3,dtype=torch.float64)
        stub=SimpleNamespace(_agent_pos=p,_agent_vel=v,_nav_targets=targets,_agent_acc=torch.zeros_like(p),
            _v_xy_max=torch.tensor([x['v_xy_max'] for x in ss]),_v_z_max=torch.tensor([x['v_z_max'] for x in ss]),
            _a_xy_max=torch.tensor([x['a_xy_max'] for x in ss]),_a_z_max=torch.tensor([x['a_z_max'] for x in ss]),
            eps=c['eps'],prior_slow_radius_xy=c['prior_slow_radius_xy'],prior_slow_radius_z=c['prior_slow_radius_z'],
            prior_kv_xy=c['prior_kv_xy'],prior_kv_z=c['prior_kv_z'])
        prior_acc=_BaseUAVEnv._compute_waypoint_prior_acc(stub).numpy()
        for i,s in enumerate(ss):
            _,result=nominal_step(p[i],v[i],targets[i],s,c)
            expected=np.asarray(v[i])+prior_acc[i]*c['dt']
            expected[:2]*=1-s['drag_xy']*c['dt'];expected[2]+=(-s['drag_z']*expected[2]+s['buoyancy_bias'])*c['dt']
            np.testing.assert_allclose(result,expected,atol=1e-8)

