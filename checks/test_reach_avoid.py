"""New algorithm regressions; no environment initialization or physics steps."""
import copy,hashlib,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from core.env.mission_env import MissionCoreEnv
from core.env.target_motion import TargetState
from checks.reach_avoid_fixtures import public_controller,learning_fixture
from chapter3_bser.reach_avoid.options import validate,INPUT_DIM,FRAME_DIM
from chapter3_bser.reach_avoid.goals import Goal,SEARCH,STANDBY,CAPTURE,VIA,encode_goal,goal_reached,next_goal,base_action
from chapter3_bser.reach_avoid.learner import first_event_cdf,bellman_events,StratifiedReplay,Sample,ReachAvoidLearner
from chapter3_bser.reach_avoid.control import project
from chapter3_bser.reach_avoid import dataset


class FirstEventLearning(unittest.TestCase):
    def test_competing_events_are_monotone_and_sum_at_most_one(self):
        logits=torch.linspace(-3.,3.,2*32*3).reshape(2,-1)
        p=first_event_cdf(logits)
        self.assertEqual(tuple(p.shape),(2,32,2));self.assertTrue(bool((p[:,1:]>=p[:,:-1]-1e-7).all()))
        self.assertTrue(bool((p.sum(-1)<=1.+1e-6).all()))
    def test_reach_first_is_not_undone_by_later_collision(self):
        logits=torch.tensor([[15.,-15.,-15.,-15.,15.,-15.]])
        p=first_event_cdf(logits)
        self.assertGreater(float(p[0,1,0]),.999);self.assertLess(float(p[0,1,1]),1e-5)
    def test_collision_wins_simultaneous_transition(self):
        target=bellman_events(torch.zeros(1,32,2),reached=torch.tensor([True]),collision=torch.tensor([True]),deadline=torch.tensor([True]))
        self.assertEqual(float(target[0,:,0].sum()),0.);self.assertEqual(float(target[0,:,1].sum()),32.)
    def test_reach_event_is_absorbing_without_discount(self):
        target=bellman_events(torch.zeros(1,32,2),reached=torch.tensor([True]),collision=torch.tensor([False]),deadline=torch.tensor([False]))
        self.assertTrue(bool((target[0,:,0]==1).all()))
    def test_no_event_first_step_and_shifted_continuation(self):
        nxt=torch.tensor([[[.2,.1],[.4,.2],[.6,.3]]])
        target=bellman_events(nxt,reached=torch.tensor([False]),collision=torch.tensor([False]),deadline=torch.tensor([False]))
        torch.testing.assert_close(target[0,0],torch.zeros(2));torch.testing.assert_close(target[0,2],nxt[0,1])
    def test_deadline_does_not_hallucinate_future_reach_or_collision(self):
        target=bellman_events(torch.ones(1,32,2),reached=torch.tensor([False]),collision=torch.tensor([False]),deadline=torch.tensor([True]))
        self.assertEqual(float(target.sum()),0.)
    def test_original_discount_fixed_but_not_applied_to_event_probability(self):
        with self.assertRaisesRegex(ValueError,'discount'):validate(dict(validate(),gamma=.99))
        self.assertEqual(validate()['gamma'],.95)


class PublicFeaturesAndGoals(unittest.TestCase):
    def test_direction_mirrors_no_longer_alias(self):
        a,p,_,_=public_controller(points=[[6.,3.6,4.]])
        b,q,_,_=public_controller(points=[[6.,2.4,4.]])
        self.assertEqual(p.features.shape,(4,FRAME_DIM*4));self.assertFalse(np.array_equal(p.features[3],q.features[3]))
    def test_goal_coordinates_condition_the_actor_input(self):
        _,p,_,_=public_controller();a=ReachAvoidLearner.inputs(p,0,Goal((8.,3.,4.)));b=ReachAvoidLearner.inputs(p,0,Goal((3.,8.,4.)))
        self.assertEqual(a.shape,(INPUT_DIM,));self.assertFalse(np.array_equal(a,b))
    def test_original_truth_derived_range_is_not_used(self):
        c,p,s,_=public_controller();d,_,_,_=public_controller();obs=np.zeros((4,28));obs[:,18]=999
        q=d.encoder.build(s,NS(executor_knows_target=False),obs,{i:(9.,3.,4.) for i in range(4)},d.geometry,d.guard)
        np.testing.assert_array_equal(p.features,q.features)
    def test_public_payload_is_gated_and_copied(self):
        target=TargetState(np.array([5.,6.,4.]),np.array([.1,0.,0.]),sample_step=2,motion_mode='constant_velocity_reflect_v1')
        runtime=NS(executor_target_assigned=False,executor_delivered_target_state=target,executor_received_target_step=2,step_count=4)
        env=MissionCoreEnv.__new__(MissionCoreEnv);env._env=runtime
        self.assertIsNone(env.get_public_delivered_target());runtime.executor_target_assigned=True
        payload=env.get_public_delivered_target();payload['payload']['position'][0]=19
        self.assertEqual(target.position[0],5.)
    def test_public_target_estimate_does_not_use_intercept_obs_vector(self):
        c,p,s,contract=public_controller();s.target_found=True;s.step=1
        target=TargetState(np.array([5.,6.,4.]),np.array([.1,0.,0.]),sample_step=0,motion_mode='constant_velocity_reflect_v1')
        delivered=dict(schema='public_delivered_target_state_v1',payload=target.to_payload(),received_step=1,physical_step=1)
        obs=np.ones((4,28))*1000
        q=c.encoder.build(s,NS(executor_knows_target=True),obs,{i:(19.,19.,7.) for i in range(4)},c.geometry,c.guard,delivered)
        np.testing.assert_allclose(q.intents[3],[5.02,6.,4.]);np.testing.assert_allclose(q.target_velocity,[.1,0.,0.])
    def test_future_public_payload_rejected(self):
        c,p,s,_=public_controller();s.target_found=True
        target=TargetState(np.array([5.,6.,4.]),np.zeros(3),sample_step=3,motion_mode='constant_velocity_reflect_v1')
        with self.assertRaisesRegex(ValueError,'future'):c.encoder.build(s,NS(executor_knows_target=True),np.zeros((4,28)),{i:(9.,3.,4.) for i in range(4)},c.geometry,c.guard,dict(schema='public_delivered_target_state_v1',payload=target.to_payload(),received_step=0,physical_step=0))
    def test_capture_is_region_not_an_obstacle_center_waypoint(self):
        _,p,_,_=public_controller();p.positions[3]=[3.45,3.,4.];p.velocities[3]=0;p.clearances[3]=.55
        index=np.ravel_multi_index((2,1,4),tuple(p.shape));p.occupied[index]=True;p.free[index]=False
        self.assertTrue(goal_reached(p,3,Goal((4.05,3.,4.),kind=CAPTURE),validate()))
        self.assertFalse(goal_reached(p,3,Goal((4.05,3.,4.),kind=SEARCH),validate()))
    def test_standby_requires_low_velocity(self):
        _,p,_,_=public_controller();g=Goal(tuple(p.positions[3]),kind=STANDBY);p.velocities[3]=[.8,0,0]
        self.assertFalse(goal_reached(p,3,g,validate()));p.velocities[3]=0;self.assertTrue(goal_reached(p,3,g,validate()))
    def test_capture_advances_from_public_estimate_only(self):
        _,p,_,_=public_controller();p.intents[3]=[7,8,4];p.target_velocity[:]=[.2,0,0]
        g=next_goal(Goal((5,6,4),(.1,0,0),CAPTURE,(5.5,6,4)),p,3)
        self.assertEqual(g.center,(7.,8.,4.));self.assertEqual(g.ref,(7.5,8.,4.))
    def test_goal_bank_excludes_completed_search_options(self):
        c,p,s,_=public_controller();q=c.encoder.build(s,NS(executor_knows_target=False),np.zeros((4,28)),{i:tuple(s.agents[i].position) for i in range(4)},c.geometry,c.guard)
        from chapter3_bser.reach_avoid.planning import candidate_goals
        goals,mask=candidate_goals(q,s,{i:tuple(s.agents[i].position) for i in range(4)},c.geometry,validate())
        self.assertFalse(mask[0,0]);self.assertTrue(mask[0,1:].any())
    def test_shared_compensated_baseline_breaks_recorded_flow_equilibrium(self):
        _,p,_,_=public_controller();p.positions[3]=[12.729618072509766,18.745105743408203,4.709079742431641];p.velocities[3]=0
        p.flow[3]=[-.1799386356,-.169631111]
        action=base_action(p,3,Goal((13.,19.,5.5)))
        net=.968*action[:2]*.8+p.flow[3]
        self.assertGreater(float(np.linalg.norm(net)),.1)


class ProjectionAndSelection(unittest.TestCase):
    def test_projection_returns_only_an_available_executed_action(self):
        a=np.array([[1.,0,0],[0,1.,0],[-1.,0,0]],np.float32)
        p,i=project([.9,0,0],a,[False,True,True]);self.assertIn(i,[1,2]);np.testing.assert_array_equal(p,a[i])
    def test_action_library_is_independent_of_network_weights(self):
        c,p,_,_=public_controller();d,q,_,_=public_controller();np.testing.assert_array_equal(p.actions,q.actions)
        self.assertEqual(p.actions.shape,(4,16,3));self.assertTrue(p.action_mask.any(-1).all())
    def test_learned_values_can_select_non_nearest_goal(self):
        _,p,_,_=public_controller();model=ReachAvoidLearner(validate());valid=np.flatnonzero(p.goal_mask[0]);self.assertGreaterEqual(len(valid),2)
        a,b=map(int,valid[:2]);scores=[dict(reach=.8,risk=.01,task=0.) for _ in range(8)];scores[b]['task']=4.
        with patch.object(model,'evaluate_queries',return_value=(scores,None)):
            chosen,_,reason=model.select_goal(p,0)
        self.assertEqual(chosen,b);self.assertEqual(reason,'learned_goal_and_residual')
    def test_risk_only_ablation_cannot_get_reach_gate_or_bonus(self):
        _,p,_,_=public_controller();model=ReachAvoidLearner(validate());valid=list(map(int,np.flatnonzero(p.goal_mask[0])));a,b=valid[:2]
        scores=[dict(reach=.01,risk=.01,task=-5.) for _ in range(8)];scores[a].update(reach=.9,task=0);scores[b]['task']=.5
        full,_=model.choose_scores(p,0,scores,'full');risk,_=model.choose_scores(p,0,scores,'safety_only')
        self.assertEqual(full,a);self.assertEqual(risk,b)
    def test_execution_and_td_policy_use_identical_projection(self):
        model,o,p=learning_fixture();model._evaluation_ready=True
        c,packet,_,_=public_controller(o);c.learner=model
        commands,goals,rows=c.act(packet)
        for i in range(4):np.testing.assert_allclose(commands[i],model.policy_command(packet,i,goals[i]),atol=1e-6)
    def test_buffer_exit_has_identical_priority_in_no_goal_repair_ablation(self):
        model,o,p=learning_fixture();model.ablation='no_goal_repair';model._evaluation_ready=True
        c,packet,_,_=public_controller(o);c.learner=model;c.ablation='no_goal_repair'
        packet.goal_mask[:,0]=False;packet.action_mask[:,9]=True;packet.checked[:,9]=True
        packet.actions[:,9]=[-.2,-.3,0.];c.bank_labels=[['BUFFER_EXIT' if j==9 else 'OTHER' for j in range(16)] for i in range(4)]
        commands,goals,rows=c.act(packet)
        for i in range(4):
            np.testing.assert_allclose(commands[i],model.policy_command(packet,i,goals[i]),atol=1e-6)
            values,_=model.evaluate_queries([(packet,i,goals[i])]);self.assertTrue(np.isfinite(values[0]['reach']))
    def test_failed_projection_is_explicitly_uncertified(self):
        c,p,_,_=public_controller(points=[[6.,3.,4.]])
        _,_,rows=c.act(p)
        if rows[3]['action_mode']=='UNCERTIFIED_BRAKE':self.assertFalse(rows[3]['model_checked'])
        else:self.assertTrue(rows[3]['model_checked'])
    def test_zero_residual_is_an_evaluation_intervention_not_weight_relabelling(self):
        model,o,p=learning_fixture();before=copy.deepcopy(model.actor.state_dict());model.intervention='zero_residual'
        np.testing.assert_array_equal(model.residual(p,0,Goal((9,3,4))),np.zeros(3))
        with self.assertRaisesRegex(ValueError,'cannot train'):model.update()
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.actor.state_dict().items()))


class ReplayAndUpdates(unittest.TestCase):
    def test_rare_success_and_collision_survive_other_data(self):
        model,o,p=learning_fixture();samples=[x for pool in model.replay.pools for x in pool]
        replay=StratifiedReplay(9);replay.add(samples[0]);replay.add(next(s for s in samples if s.collision))
        for _ in range(100):replay.add(replace(samples[0],positive_behavior=False,collision=False))
        self.assertEqual(replay.counts,dict(reach=1,collision=1,other=3))
    def test_stratified_weights_correct_retained_distribution(self):
        model,o,p=learning_fixture();samples=[x for pool in model.replay.pools for x in pool];replay=StratifiedReplay(9)
        replay.add(samples[0]);replay.add(next(s for s in samples if s.collision))
        for _ in range(3):replay.add(replace(samples[0],positive_behavior=False,collision=False))
        rows,w=replay.draw(6,np.random.default_rng(1))
        self.assertAlmostEqual(float(sum(x for s,x in zip(rows,w) if s.collision)/w.sum()),.2,places=6)
    def test_initialization_preserves_original_torch_rng(self):
        before=torch.random.get_rng_state().clone();ReachAvoidLearner(validate());self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
    def test_four_joint_updates_change_actor_and_critics_without_reward_edit(self):
        model,o,p=learning_fixture();before=[x.detach().clone() for x in model.actor.parameters()]
        old_critic=[x.detach().clone() for x in model.critics.parameters()]
        for _ in range(4):
            metric=model.update();self.assertTrue(np.isfinite(metric['event_loss_and_task_loss']));self.assertFalse(metric['task_reward_changed'])
        self.assertEqual(model.updates,4);self.assertTrue(model.ready)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.actor.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old_critic,model.critics.parameters())))
    def test_no_collision_support_cannot_become_ready(self):
        model,o,p=learning_fixture();model.updates=4;model.replay.pools[1].clear();self.assertFalse(model.ready)
    def test_hindsight_never_receives_original_task_return_labels(self):
        model,o,p=learning_fixture();model.replay=StratifiedReplay(o['replay_size']);nxt=copy.deepcopy(p);nxt.step=1;nxt.positions[0]+=[2.,0.,0.]
        row=dict(packet=p,next_packet=nxt,goals=p.goals[:,0].copy(),actual_actions=np.zeros((4,3),np.float32),reward=123.,done=True,team_collision=False,rows=[dict(projection_changed=False)]*4)
        model.add_episode([row]);counterfactual=[s for pool in model.replay.pools for s in pool if not s.task_label]
        self.assertTrue(counterfactual);self.assertTrue(any(s.positive_behavior for s in counterfactual))
    def test_old_policy_cannot_load_under_new_identity(self):
        model=ReachAvoidLearner(validate())
        with self.assertRaisesRegex(ValueError,'identity'):model.load_for_evaluation(dict(input_dim=192,updates=100))


class FactualDatasetAndAdmission(unittest.TestCase):
    def record(self):
        _,p,_,_=public_controller();nxt=copy.deepcopy(p);nxt.step=1
        return dict(packet=p,next_packet=nxt,goals=p.goals[:,0],actual_actions=np.full((4,3),-.2,np.float32),
            reward=.125,done=True,team_collision=False,rows=[dict(projection_changed=True)]*4)
    def test_dataset_roundtrip_keeps_actual_controls_and_public_direction_history(self):
        record=self.record();o=validate()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'factual.npz';receipt=dataset.save(path,[record]);rows=dataset.load_file(path,receipt,o)
        np.testing.assert_array_equal(rows[0]['actual_actions'],record['actual_actions'])
        np.testing.assert_array_equal(rows[0]['packet'].features,record['packet'].features)
        self.assertEqual(rows[0]['reward'],.125)
    def test_modified_dataset_fails_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'factual.npz';receipt=dataset.save(path,[self.record()]);path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):dataset.load_file(path,receipt,validate())
    def test_incomplete_episode_is_not_training_evidence(self):
        record=self.record();record['done']=False
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'factual.npz';receipt=dataset.save(path,[record])
            with self.assertRaisesRegex(ValueError,'termination'):dataset.load_file(path,receipt,validate())
    def test_commands_default_to_dry_run(self):
        from chapter3_bser.reach_avoid.workflow import main
        with patch('chapter3_bser.reach_avoid.workflow.execute',side_effect=AssertionError('must not execute')):
            for command in ('collect','train','evaluate'):
                self.assertFalse(main([command,'--config','missing','--scenarios','missing','--output','missing','--validation','missing','--episodes','1'])['executed'])
    def test_linux_validation_required_before_research(self):
        from chapter3_bser.reach_avoid.workflow import admission
        with patch('chapter3_bser.reach_avoid.workflow.verify_reviewed_changes',return_value=dict(profile='active_reach_avoid_v1')),patch('chapter3_bser.reach_avoid.workflow.research_readiness'),patch('chapter3_bser.learning.map_suite.platform.system',return_value='Windows'):
            with self.assertRaisesRegex(ValueError,'Linux'):admission('missing')
