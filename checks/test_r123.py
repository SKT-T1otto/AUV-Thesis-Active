"""R123 contracts: synthetic public fixtures, no native environment/restore."""
import copy
from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from chapter3_bser.path_option.options import DEFAULTS
from chapter3_bser.path_option.path_context import Route
from chapter3_bser.r123.options import ARMS,CONTROL,switches
from chapter3_bser.r123.control import conditional_banks
from chapter3_bser.r123.lifecycle import Lifecycle,ProgressManager,arrived
from chapter3_bser.r123.learner import Learner,option_return
from chapter3_bser.r123.networks import tensor_batch
from chapter3_bser.r123.runtime import PathOptionRuntime
from checks.test_path_option import fixture as old_fixture


def fixture():
    r=old_fixture();r['actions']=np.repeat(r['actions'][None],8,axis=0)
    r['action_mask']=np.repeat(r['action_mask'][None],8,axis=0)
    r['forced']=np.full(8,-1);return r


def runtime_fixture(arm='full'):
    from checks.reach_avoid_fixtures import public_controller
    from chapter3_bser.reach_avoid.features import Encoder
    from chapter3_bser.reach_avoid.scan_geometry import RaySubcells
    controller,_,state,contract=public_controller()
    state.occupancy.occupancy_probability=np.ones(800)*.1
    r=PathOptionRuntime.__new__(PathOptionRuntime)
    r.arm=arm;r.switches=switches(arm);r.o=dict(DEFAULTS)
    r.lifecycle=Lifecycle(CONTROL);r.lifecycle_events=r.lifecycle.events
    r.manager=ProgressManager(r.o,r.lifecycle,r.switches['r2'])
    r.guard=controller.guard;r.subcells=RaySubcells(contract['bounds']);r.encoder=Encoder(dict(history=16),400)
    r.faults=[];r._cached=None;r._alternatives={};r._last_candidates={}
    r._candidate_goal_ids={};r._requested_goal_ids=set();r._candidates={}
    r._release_requests=set();r._last_request=-10000;r._request_refresh=False;r.intervention='none'
    r.observations=np.zeros((4,28));r.learner=None
    r.guidance=NS(agent_assignments=tuple(NS(agent_id=i,final_waypoint=(9.,3.,4.)) for i in range(4)))
    task=NS(step=0,executor_knows_target=False)
    r.env=NS(get_task_state=lambda:task,read_public_events=lambda cursor:(),get_public_delivered_target=lambda:None)
    def query(i,point,s):
        p=np.array(s.agents[i].position)
        return Route([p,p+[0.,1.,0.],point],s.map_revision,s.step,True,
                     distance=float(np.linalg.norm(point-p)),estimated_seconds=5.)
    r._query=query
    return r,state,task


def observe(r,state):
    with patch('chapter3_bser.r123.runtime.extract_planning_state',return_value=state):
        return r.observe()


class R123Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_ablation_switches_change_exactly_one_component(self):
        full=switches('full')
        for arm in ARMS[1:]:
            s=switches(arm);self.assertEqual([k for k in full if s[k]!=full[k]],[arm[3:]])

    def test_network_shapes_and_goal_conditional_masks(self):
        row=fixture();row['action_mask'][2,:]=False;row['action_mask'][2,4]=True
        model=Learner()
        with torch.no_grad():low,*_=model.distributions(tensor_batch([row]),[row])
        self.assertEqual(low.shape,(1,8,16));self.assertEqual(float(low[0,2,4]),1.)

    def test_forced_action_is_per_goal(self):
        row=fixture();row['forced'][1]=8
        m=Learner()
        with torch.no_grad():low,*_=m.distributions(tensor_batch([row]),[row])
        self.assertEqual(float(low[0,1,8]),1.);self.assertLess(float(low[0,0,8]),1.)

    def test_r1_masks_probes_and_keeps_bounded_residuals(self):
        r,state,_=runtime_fixture();rows=observe(r,state)
        for row in rows:
            self.assertFalse(row['action_mask'][:,10:].any())
            for bank in row['actions']:
                self.assertTrue((np.abs(bank[1:7]-bank[0])<=.150001).all())

    def test_no_r1_retains_original_shared_library(self):
        r,state,_=runtime_fixture('no_r1');rows=observe(r,state)
        self.assertTrue(any(row['action_mask'][:,10:].any() for row in rows))
        np.testing.assert_array_equal(rows[0]['actions'][0],rows[0]['actions'][7])

    def test_no_r2_is_only_arm_with_learned_beta(self):
        for arm in ARMS:
            model=Learner(arm=arm)
            self.assertEqual(model.variant,'A' if arm=='no_r2' else 'C')

    def test_r2_cannot_terminate_continuing_task_at_ten_steps(self):
        r,state,task=runtime_fixture();observe(r,state)
        state.step=10;task.step=10;r._cached=None
        rows=observe(r,state)
        for row in rows:
            if row['reason']=='CONTINUE':
                self.assertFalse(row['forced_terminate']);self.assertFalse(row['beta_allowed'])

    def test_progress_timer_not_reset_by_recommitting_same_goal(self):
        l=Lifecycle(CONTROL);l.accept(0,[9,3,4],0,0,[3,3,4])
        m=ProgressManager(DEFAULTS,l,True);m.public_pose[0]=(True,2.)
        for step in range(31):
            m.commit(0,[9,3,4],[9,3,4],[9,3,4],step,0,Route([[3,3,4],[9,3,4]],1,0,True),reason='REPEAT')
            reason=m.status(0,np.array([3.,3.,4.]),np.zeros(3),0,step,True)
        self.assertEqual(reason,'STALLED');self.assertEqual(l.tasks[0].progress_step,0)

    def test_arrival_requires_same_public_free_and_clearance_as_label(self):
        for free,clear in [(False,2.),(True,.52)]:
            self.assertFalse(arrived(np.ones(3),np.zeros(3),np.ones(3),0,free,clear))
        self.assertTrue(arrived(np.ones(3),np.zeros(3),np.ones(3),0,True,.53))
        self.assertFalse(arrived(np.ones(3),np.ones(3),np.ones(3),1,True,2.))

    def test_r3_recovery_arrival_returns_to_original_task(self):
        r,state,task=runtime_fixture();observe(r,state)
        c=r.manager.current[0];original=c.task.copy();point=np.array(state.agents[0].position)
        r.lifecycle.enter_recovery(0,0,point)
        r.manager.commit(0,original,original,point,0,0,Route([point],1,0,True),reason='RECOVER')
        state.step=1;task.step=1;r._cached=None;rows=observe(r,state)
        self.assertEqual(r.manager.current[0].mode,'TASK')
        np.testing.assert_array_equal(r.manager.current[0].endpoint,original)
        self.assertTrue(any(e['event']=='RECOVERY_RETURN' for e in r.lifecycle.events))

    def test_r3_timeout_is_not_extended_by_goal_recommit(self):
        r,state,task=runtime_fixture();observe(r,state)
        c=r.manager.current[0];point=c.task+[0.,2.,0.]
        r.lifecycle.enter_recovery(0,0,point)
        r.manager.commit(0,c.original,c.task,point,149,0,Route([point],1,149,True),reason='REPEAT')
        state.step=150;task.step=150;r._cached=None
        row=observe(r,state)[0]
        self.assertEqual(row['reason'],'WAIT_TASK')
        self.assertEqual(r.lifecycle.tasks[0].release_reason,'RECOVERY_TIME_EXHAUSTED')

    def test_task_release_waits_without_fake_success_or_probe(self):
        r,state,task=runtime_fixture();observe(r,state)
        r.lifecycle.release(0,0,'TEST');r._cached=None
        row=observe(r,state)[0]
        self.assertEqual(row['reason'],'WAIT_TASK');self.assertTrue((row['forced']==8).all())
        self.assertFalse(row['forced_terminate']);self.assertIn(0,r._release_requests)

    def test_new_d2_task_is_adopted_only_after_release(self):
        r,state,task=runtime_fixture();observe(r,state);old=r.manager.current[0].task.copy()
        r.guidance.agent_assignments[0].final_waypoint=(15.,3.,4.)
        state.step=1;task.step=1;r._cached=None;observe(r,state)
        np.testing.assert_array_equal(r.manager.current[0].task,old)
        r.lifecycle.release(0,1,'TEST');state.step=2;task.step=2;r._cached=None;observe(r,state)
        np.testing.assert_array_equal(r.manager.current[0].task,[15.,3.,4.])

    def test_discovery_and_handoff_preempt_recovery(self):
        r,state,task=runtime_fixture();observe(r,state)
        from core.env.target_motion import TargetState
        payload=TargetState([12.,3.,4.],[.3,0,0],1,'constant_velocity_reflect_v1',state_schema='moving_target_state_v1',obstacle_layout_id='custom_aabb_v1').to_payload()
        state.step=1;state.target_found=True;task.step=1;task.executor_knows_target=True
        r.env.get_public_delivered_target=lambda:dict(schema='public_delivered_target_state_v1',physical_step=1,received_step=1,payload=payload)
        r._cached=None;rows=observe(r,state)
        self.assertEqual([x['phase'] for x in rows],[3,3,3,2])
        for row in rows[:3]:
            self.assertTrue((row['forced']==8).all());self.assertTrue((row['actions'][:,8]==0).all())
        self.assertEqual(r.lifecycle.tasks[3].phase,2)

    def test_action_sampling_uses_selected_goals_own_bank(self):
        row=fixture();row['beta_allowed']=False;row['forced_terminate']=True
        row['goal_mask'][:]=False;row['goal_mask'][5]=True
        row['forced'][5]=8
        model=Learner();choice=model.decide(row)
        self.assertEqual((choice['goal'],choice['action']),(5,8))

    def test_executor_retries_root_without_resetting_recovery_budget(self):
        r,state,task=runtime_fixture();observe(r,state)
        from core.env.target_motion import TargetState
        payload=TargetState([12.,3.,4.],[.3,0,0],1,'constant_velocity_reflect_v1',state_schema='moving_target_state_v1',obstacle_layout_id='custom_aabb_v1').to_payload()
        state.step=1;state.target_found=True;task.step=1;task.executor_knows_target=True
        r.env.get_public_delivered_target=lambda:dict(schema='public_delivered_target_state_v1',physical_step=1,received_step=1,payload=payload)
        r._cached=None;observe(r,state)
        root=r.lifecycle.tasks[3];serial=root.identity;root.recoveries=2
        r.lifecycle.release(3,1,'RECOVERY_EXHAUSTED')
        state.step=2;task.step=2;r._cached=None
        r.env.get_public_delivered_target=lambda:dict(schema='public_delivered_target_state_v1',physical_step=state.step,received_step=1,payload=payload)
        rows=observe(r,state)
        self.assertEqual(r.lifecycle.tasks[3].identity,serial)
        self.assertEqual(root.recoveries,2);self.assertTrue(root.root_only)
        self.assertEqual(rows[3]['reason'],'TASK_ONLY');self.assertFalse(rows[3]['forced_terminate'])
        np.testing.assert_array_equal(r.manager.current[3].endpoint,r.manager.current[3].task)

    def test_actions_execute_selected_goal_bank_and_install_same_route(self):
        from chapter3_bser.integration.control_context import AgentAssignmentContextV1 as A,ExecutorAssignmentContextV1 as E,BSERControlContextV1 as C
        r,state,_=runtime_fixture();rows=observe(r,state)
        assignments=tuple(A(i,'searcher' if i<3 else 'executor','search','test',(9.,3.,4.),(),(9.,3.,4.),(9.,3.,4.),False,True,False) for i in range(4))
        r.guidance=C('bser.control_context.v1','v','h',0,'SEARCH',assignments,E(3,'test',(9.,3.,4.),(),(9.,3.,4.),(9.,3.,4.),False,True,False),'TEST')
        installed=[];r.env.install_guidance=installed.append;r._explore=False
        r.learner=NS(decide=lambda row,explore:dict(goal=1,action=0,terminate=True,beta=1.,forced=-1,reach=None,risk=None,task=None))
        with patch('chapter3_bser.r123.runtime.RepairedMissionRuntime._actions_for_step',return_value=None):
            actual=r._actions_for_step(torch.zeros((4,3)))
        np.testing.assert_array_equal(actual.numpy(),np.array([row['actions'][1,0] for row in rows]))
        for i,item in enumerate(installed[0].agent_assignments):
            np.testing.assert_array_equal(item.final_waypoint,r._candidates[i][1][0])

    def test_evaluation_preserves_all_learning_and_rng_state(self):
        from chapter3_bser.r123.workflow import evaluate,learning_state
        m=Learner();before=learning_state(m)
        py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state();lr=copy.deepcopy(m.rng.bit_generator.state)
        def roll(*a,**kw):
            random.random();np.random.random();torch.rand(1);m.rng.random()
            return {},dict(success=0,collision=0,timeout=1),True
        with tempfile.TemporaryDirectory() as folder:
            with patch('chapter3_bser.r123.workflow.one_episode',side_effect=roll),patch('chapter3_bser.r123.workflow.identity',return_value={'test':True}),patch('chapter3_bser.r123.workflow.scenario_fingerprint',return_value='fixture'):
                evaluate({},[dict(scenario_id='a')],DEFAULTS,m,Path(folder)/'eval',arm='full')
        self.assertEqual(learning_state(m),before);self.assertEqual(py,random.getstate())
        np.testing.assert_equal(nr,np.random.get_state());self.assertTrue(torch.equal(tr,torch.get_rng_state()))
        self.assertEqual(lr,m.rng.bit_generator.state)

    def test_budget_cutoff_does_not_fake_environment_done(self):
        from chapter3_bser.r123.workflow import one_episode
        runtime=NS(terminal=False,transitions=[],close=lambda:None,summary=lambda:{'path_option':{'trace':[]}})
        def advance(*a,**kw):runtime.transitions.extend(dict(done=False) for _ in range(4))
        runtime.advance=advance;model=NS(add=lambda t:None);updates=[]
        with patch('chapter3_bser.r123.workflow.PathOptionRuntime',return_value=runtime):
            _,metric,complete=one_episode(dict(max_steps=400),dict(scenario_id='fixture'),DEFAULTS,model,arm='full',seed=1,episode_id=1,training=True,remaining_steps=3,update_callback=lambda m:updates.append(1))
        self.assertFalse(complete);self.assertTrue(metric['budget_censored'])
        self.assertFalse(any(t['done'] for t in runtime.transitions));self.assertEqual(len(updates),3)

    def test_all_arms_four_complete_updates_finite_and_beta_gradient_disabled(self):
        for arm in ARMS:
            o=dict(DEFAULTS,batch_size=2,warmup=2)
            m=Learner(o,arm)
            for k in range(4):
                s=fixture();s.update(phase=0)
                m.add(dict(state=s,next=copy.deepcopy(s),goal_index=k,actual_action=s['actions'][k,0],reward=.1,
                    done=k==0,reached=k==1,collision=k==2,event_deadline=False,event_target=np.ones(3),policy_executed=True,decision=True))
            for _ in range(4):
                loss=m.update();self.assertTrue(np.isfinite(loss['actor_loss']))
            self.assertEqual(m.updates,4)
            if arm!='no_r2':
                self.assertTrue(all(p.grad is None or not bool(p.grad.any()) for p in m.policy.termination.parameters()))

    def test_task_discount_unchanged(self):
        result=option_return(torch.tensor([1.]),torch.tensor([0.]),torch.tensor([2.]),torch.tensor([4.]),torch.tensor([.25]))
        self.assertAlmostEqual(float(result),3.375)

    def test_preview_does_not_generate_files_or_environment(self):
        from tools.r123_experiment import main
        with patch('tools.r123_experiment.prepare',side_effect=AssertionError('mutation forbidden')):
            self.assertFalse(main(['prepare','--output','unused'])['executed'])


if __name__=='__main__':unittest.main()
