"""Regressions for observed route resets and fixed-waypoint feedforward."""
import copy
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np

from checks.test_r123_governed import packet, geometry
from checks.test_r123_repair import fixture
from chapter3_bser.path_option.path_context import Route
from chapter3_bser.r123_governed.control import conditional_banks as parent_banks
from chapter3_bser.r123_governed.runtime import RuleRuntime as ParentRuntime
from chapter3_bser.r123_tracking.control import conditional_banks, governed_base
from chapter3_bser.r123_tracking.progress import carry_progress, tracking_velocity
from chapter3_bser.r123_tracking.runtime import RuleRuntime, ARMS
from chapter3_bser.reach_avoid.options import SPECS, CONTRACT
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.options import validate
from chapter3_bser.safe_rl.dynamics import integrate, limits


def route(points,index=0,reachable=True):
    return Route(points,1,0,reachable,index=index)


class TrackingContracts(unittest.TestCase):
    def test_observed_backward_start_vertex_is_not_reintroduced(self):
        old=route([[3.,3.,2.5],[5.,5.,2.5],[15.,15.,4.5]],1)
        p=np.array([3.457,3.652,2.5])
        fresh=route([p,[3.,3.,2.5],[5.,5.,2.5],[15.,15.,4.5]])
        report=carry_progress(old,fresh,p,geometry(),.75)
        self.assertTrue(report['advanced']);self.assertEqual(fresh.index,2)
        np.testing.assert_equal(fresh.tracking(p),[5.,5.,2.5])
        json.dumps(report,allow_nan=False)

    def test_fresh_geometry_veto_retains_required_old_vertex(self):
        old=route([[3.,3.,2.5],[5.,5.,2.5]],1);p=np.array([3.457,3.652,2.5])
        fresh=route([p,[3.,3.,2.5],[5.,5.,2.5]])
        geo=geometry();geo.segment=lambda *a:False
        report=carry_progress(old,fresh,p,geo,.75)
        self.assertEqual(report['reason'],'fresh_geometry_veto');self.assertEqual(fresh.index,1)

    def test_cannot_skip_new_obstacle_detour_or_uncompleted_vertex(self):
        p=np.array([3.457,3.652,2.5]);old=route([[3.,3.,2.5],[5.,5.,2.5]],1)
        for points in ([p,[3.,3.,2.5],[4.,2.,2.5],[5.,5.,2.5]],
                       [p,[3.,3.,2.5],[7.,7.,2.5]],
                       [p,[3.,3.,2.5],[5.,5.,2.5]]):
            fresh=route(points);prior=copy.deepcopy(old)
            if len(points)==3:prior.index=0
            self.assertFalse(carry_progress(prior,fresh,p,geometry(),.75)['advanced'])
            self.assertEqual(fresh.index,1)

    def test_unreachable_routes_are_not_upgraded(self):
        for old_ok,new_ok in ((False,True),(True,False)):
            old=route([[3.,3.,2.5],[5.,5.,2.5]],1,old_ok)
            fresh=route([[3.5,3.7,2.5],[3.,3.,2.5],[5.,5.,2.5]],reachable=new_ok)
            self.assertFalse(carry_progress(old,fresh,[3.5,3.7,2.5],geometry(),.75)['advanced'])
            self.assertEqual(fresh.reachable,new_ok)

    def test_fixed_waypoint_velocity_zero_terminal_moving_target_preserved(self):
        p=packet();p.phases[3]=2;p.target_velocity=np.array([.28,-.15,.20])
        r=route([[3.,3.,4.],[5.,5.,4.],[9.,7.,5.]],1)
        np.testing.assert_equal(tracking_velocity(p,3,r,[9.,7.,5.],True),[0.,0.,0.])
        np.testing.assert_equal(tracking_velocity(p,3,r,[9.,7.,5.],False),p.target_velocity)
        r.index=2
        np.testing.assert_equal(tracking_velocity(p,3,r,[9.,7.,5.],True),p.target_velocity)
        np.testing.assert_equal(tracking_velocity(p,3,r,[8.,7.,5.],True),[0.,0.,0.])
        p.phases[3]=3
        np.testing.assert_equal(tracking_velocity(p,3,r,[9.,7.,5.],True),[0.,0.,0.])

    def test_old_static_waypoint_equilibrium_is_no_longer_stationary(self):
        p=packet();i=3;p.phases[i]=2;p.target_velocity=np.array([.2806076,-.1531325,.2025974])
        caps=(.85717428,.75,1.)
        track=np.array([13.,9.,2.5])
        offset=np.r_[2.4*p.target_velocity[:2]/caps[0],1.2*p.target_velocity[2]/caps[1]]
        p.positions[i]=track+offset
        self.assertGreater(np.linalg.norm(offset),.75)
        old=governed_base(p,i,track,caps,p.target_velocity)
        new=governed_base(p,i,track,caps,np.zeros(3))
        oldv=integrate(p.positions[i],p.velocities[i],old*limits(SPECS[i]),SPECS[i],.2)[1]
        newv=integrate(p.positions[i],p.velocities[i],new*limits(SPECS[i]),SPECS[i],.2)[1]
        self.assertLess(np.linalg.norm(oldv),1e-8)
        self.assertGreater(np.dot(newv,track-p.positions[i]),0.)

    def test_disabled_fixes_match_parent_actions_masks_forces_exactly(self):
        rng=np.random.default_rng(91);g=SafetyController(SPECS,CONTRACT,validate())
        for phase in (0,1,2,3):
            for i in range(4):
                p=packet();p.phases[:]=phase;p.velocities=rng.uniform(-.2,.2,(4,3));p.flow=rng.uniform(-.1,.1,(4,2));p.target_velocity=rng.uniform(-.3,.3,3)
                tracks=rng.uniform(2.5,4.,(8,3))
                with patch.object(g,'_caps',return_value=(.4,.3,1.)):
                    old=parent_banks(p,i,tracks,geometry(),g)
                    new=conditional_banks(p,i,tracks,geometry(),g)
                for a,b in zip(old[:4],new[:4]):np.testing.assert_array_equal(a,b)

    def test_fixed_feedforward_bank_keeps_residuals_safety_and_real_authority(self):
        p=packet();p.phases[3]=2;p.target_velocity=np.array([.28,-.15,.2]);g=SafetyController(SPECS,CONTRACT,validate())
        with patch.object(g,'_caps',return_value=(.1,.07,.1)):
            a,m,forced,emergency,d=conditional_banks(p,3,[[9.,3.,4.]]*8,geometry(),g,target_velocities=[np.zeros(3)]*8)
        self.assertTrue(np.all(np.abs(a)<=1.));self.assertFalse(m[:,10:].any())
        self.assertTrue(np.all(np.abs(a[:,1:7]-a[:,0,None])<=.150001))
        self.assertEqual(d[0]['tracking_velocity'],[0.,0.,0.])
        v=integrate(p.positions[3],p.velocities[3],a[0,0]*limits(SPECS[3]),SPECS[3],.2)[1]
        self.assertLessEqual(np.linalg.norm(v[:2]),.100001);self.assertLessEqual(abs(v[2]),.070001)

    def fixture(self,arm):
        r,s,t=fixture('no_recovery');r.__class__=RuleRuntime
        r.tracking_arm=arm;r.tracking_switches=dict(zip(('path','feedforward'),ARMS[arm]));r.route_events=[]
        return r,s,t

    def observe(self,r,s):
        with patch('chapter3_bser.r123_tracking.runtime.extract_planning_state',return_value=s):return r.observe()

    def test_all_arms_keep_no_recovery_and_frozen_actions(self):
        for arm in ARMS:
            r,s,t=self.fixture(arm);self.observe(r,s);s.target_found=True;s.step=200;t.step=200;r._cached=None
            rows=self.observe(r,s)
            for row in rows[:3]:
                self.assertEqual(row['reason'],'FROZEN');self.assertFalse(row['rule_terminate'])
                self.assertTrue((row['actions'][:,8]==0).all())
            self.assertFalse(any(e['event']=='RECOVERY_ENTER' for e in r.lifecycle.events))

    def test_route_carry_is_scoped_to_same_live_task(self):
        r,s,t=self.fixture('both');self.observe(r,s)
        del r._query  # remove the inherited test fixture stub, exercise runtime wrapper
        c=r.manager.current[0];c.route=route([[3.,3.,4.],[9.,3.,4.]],1)
        c.route.tracking_owner=(0,r.lifecycle.task_id(0)+1)
        with patch.object(ParentRuntime,'_query',return_value=route([[3.,3.,4.],[9.,3.,4.]])):
            r._query(0,c.endpoint,s)
        self.assertFalse(r.route_events)
        c.route.tracking_owner=(0,r.lifecycle.task_id(0));r.lifecycle.tasks[0].released=True
        with patch.object(ParentRuntime,'_query',return_value=route([[3.,3.,4.],[9.,3.,4.]])):
            r._query(0,c.endpoint,s)
        self.assertFalse(r.route_events)

    def test_runtime_requery_keeps_carried_progress_on_successive_queries(self):
        r,s,t=self.fixture('both');self.observe(r,s);del r._query
        p=np.array([3.457,3.652,4.]);s.agents[0].position=tuple(p)
        c=r.manager.current[0];c.route=route([[3.,3.,4.],[5.,5.,4.],c.endpoint],1)
        c.route.tracking_owner=(0,r.lifecycle.task_id(0));r._geometry=geometry()
        for step in (100,110,120):
            s.step=step
            fresh=route([p,[3.,3.,4.],[5.,5.,4.],c.endpoint])
            with patch.object(ParentRuntime,'_query',return_value=fresh):
                c.route=r._query(0,c.endpoint,s)
            self.assertEqual(c.route.index,2)
            self.assertEqual(c.route.tracking_owner,(0,r.lifecycle.task_id(0)))
        self.assertEqual(sum(e['advanced'] for e in r.route_events),3)

    def test_runtime_feedforward_switch_applies_to_static_capture_path(self):
        from core.env.target_motion import TargetState
        for arm in ARMS:
            r,s,t=self.fixture(arm);s.target_found=True;t.executor_knows_target=True
            payload=TargetState([12.,3.,4.],[.28,-.15,.2],0,'constant_velocity_reflect_v1').to_payload()
            r.env.get_public_delivered_target=lambda:dict(schema='public_delivered_target_state_v1',
                physical_step=0,received_step=0,payload=payload)
            row=self.observe(r,s)[3]
            diagnostic=row['action_diagnostics'][0]
            if r.tracking_switches['feedforward']:
                np.testing.assert_equal(diagnostic['tracking_velocity'],np.zeros(3))
            else:
                np.testing.assert_equal(diagnostic['tracking_velocity'],[.28,-.15,.2])

    def test_invalid_arm_and_active_recovery_rejected(self):
        with self.assertRaises(ValueError):RuleRuntime({},None,tracking_arm='unknown')
        with self.assertRaises(ValueError):RuleRuntime({},None,variant='repaired')

    def test_baseline_runtime_matches_parent_observation(self):
        old,s,t=fixture('no_recovery');old.__class__=ParentRuntime
        with patch('chapter3_bser.r123_governed.runtime.extract_planning_state',return_value=s):
            expected=old.observe()
        new,state,task=self.fixture('baseline');actual=self.observe(new,state)
        for a,b in zip(expected,actual):
            for key in ('actions','action_mask','goal_mask','forced','rule_goal','rule_terminate','reason'):
                np.testing.assert_equal(a[key],b[key])

    def test_nullable_metrics_do_not_break_summary_or_become_false_zeroes(self):
        from chapter3_bser.r123_tracking.workflow import aggregate
        rows=[dict(scenario_id='a',success=0,found_step=None,executor_min_distance=None),
              dict(scenario_id='b',success=1,found_step=186,executor_min_distance=.59)]
        value=aggregate(rows)
        self.assertEqual(value['found_step'],186);self.assertEqual(value['present_counts']['found_step'],1)
        self.assertIsNone(aggregate(rows[:1])['found_step']);json.dumps(value,allow_nan=False)

    def test_experiment_dry_run_creates_nothing(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from tools.r123_tracking_experiment import main
        with redirect_stdout(StringIO()):
            value=main(['prepare','--previous','does-not-exist','--output','does-not-exist'])
        self.assertFalse(value['executed'])


if __name__=='__main__':unittest.main()
