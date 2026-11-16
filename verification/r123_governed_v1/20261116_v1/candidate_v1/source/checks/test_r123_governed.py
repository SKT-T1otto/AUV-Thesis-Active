import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
from checks.test_r123_repair import fixture
from chapter3_bser.r123_governed.control import conditional_banks,governed_base
from chapter3_bser.r123.control import conditional_banks as old_banks
from chapter3_bser.r123_governed.runtime import RuleRuntime
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.options import validate
from chapter3_bser.safe_rl.dynamics import limits,integrate
from chapter3_bser.reach_avoid.options import SPECS,CONTRACT


def packet():return NS(positions=np.array([[3.,3.,4.]]*4),velocities=np.zeros((4,3)),flow=np.zeros((4,2)),phases=np.zeros(4,int),target_velocity=np.zeros(3))
def geometry():return NS(distance=lambda p:10.,padding=.52,segment=lambda *a,**k:True)
def observe(r,s):
    with patch('chapter3_bser.r123_governed.runtime.extract_planning_state',return_value=s):return r.observe()
def tick(r,s,t,n):
    s.step=n;t.step=n;r._cached=None
    return observe(r,s)


class GovernedContracts(unittest.TestCase):
    def test_positive_cap_missing_action_regression(self):
        p=packet();g=SafetyController(SPECS,CONTRACT,validate());tracks=[np.array([9.,3.,4.])]*8
        with patch.object(g,'_caps',return_value=(.1,.1,.04)):
            old=old_banks(p,0,tracks,geometry(),g,True)
            a,m,f,e,d=conditional_banks(p,0,tracks,geometry(),g)
        self.assertEqual(np.flatnonzero(old[1][0]).tolist(),[8])
        self.assertTrue(m[0,0]);self.assertFalse(m[0,8]);self.assertEqual(f[0],-1)
        nv=integrate(p.positions[0],p.velocities[0],a[0,0]*limits(SPECS[0]),SPECS[0],.2)[1]
        self.assertTrue(0<np.linalg.norm(nv[:2])<=.100001)
        self.assertFalse(d[0]['navigation_blocked'])

    def test_zero_cap_does_not_bypass_safety_or_claim_navigation(self):
        p=packet();g=SafetyController(SPECS,CONTRACT,validate())
        with patch.object(g,'_caps',return_value=(0.,0.,0.)):
            a,m,f,e,d=conditional_banks(p,0,[[9.,3.,4.]]*8,geometry(),g)
        self.assertEqual(np.flatnonzero(m[0]).tolist(),[8]);self.assertEqual(f[0],8)
        self.assertTrue(d[0]['navigation_blocked']);self.assertFalse(e[0])

    def test_geometry_veto_not_relaxed(self):
        p=packet();g=SafetyController(SPECS,CONTRACT,validate());geo=geometry();geo.segment=lambda *a,**k:False
        with patch.object(g,'_caps',return_value=(.1,.1,.04)):
            _,m,f,e,d=conditional_banks(p,0,[[9.,3.,4.]]*8,geo,g)
        self.assertTrue(e[0]);self.assertEqual(f[0],8)
        self.assertTrue(any(x['reason']=='motion_or_backup_geometry' for x in d[0]['candidate_checks']))

    def test_physical_authority_and_next_speed_with_flow_and_target_motion(self):
        p=packet();p.flow[:]=[.12,-.08];p.phases[:]=2;p.target_velocity=np.array([.7,.3,.1])
        for i,s in enumerate(SPECS):
            a=governed_base(p,i,[9.,3.,4.],(.1,.07,.1))
            self.assertTrue((np.abs(a)<=1).all())
            nv=integrate(p.positions[i],p.velocities[i],a*limits(s),s,.2,p.flow[i])[1]
            self.assertLessEqual(np.linalg.norm(nv[:2]),.100001);self.assertLessEqual(abs(nv[2]),.070001)

    def test_overspeed_is_braked_with_real_limits(self):
        p=packet();p.velocities[0]=[1.,0.,0.]
        a=governed_base(p,0,[9.,3.,4.],(.1,.1,.1))
        nv=integrate(p.positions[0],p.velocities[0],a*limits(SPECS[0]),SPECS[0],.2)[1]
        self.assertTrue((np.abs(a)<=1).all());self.assertTrue(.1<nv[0]<1.)

    def test_trace_input_diagnostics_masks_and_residual_bound(self):
        r,s,t=fixture('no_recovery');r.__class__=RuleRuntime
        rows=observe(r,s)
        for row in rows:
            self.assertIn('action_diagnostics',row);self.assertEqual(row['assignment_kind'],'search' if row['phase']==0 else 'executor')
            self.assertTrue((np.abs(row['actions'][:,1:7]-row['actions'][:,0,None])<=.150001).all())
            self.assertFalse(row['action_mask'][:,10:].any())
        s.target_found=True
        rows=tick(r,s,t,200)
        for row in rows[:3]:
            self.assertFalse(row['rule_terminate']);self.assertTrue((row['actions'][:,8]==0).all())

    def test_no_learning_or_active_recovery(self):
        with self.assertRaises(ValueError):RuleRuntime({},None,learner=object())
        with self.assertRaises(ValueError):RuleRuntime({},None,variant='repaired')


if __name__=='__main__':unittest.main()
