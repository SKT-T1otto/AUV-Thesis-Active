"""Observer equivalence and public failure classification. No native envs."""
from types import SimpleNamespace as NS
import unittest
import numpy as np
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.options import validate
from chapter3_bser.reach_avoid.scan_geometry import RaySubcells,ScanRefinedGeometry,ray_cells
from chapter3_bser.r123_task_sync.diagnostics import ProbeGeometry,FailureObserver
from checks.reach_avoid_fixtures import public_controller


class DiagnosticsContracts(unittest.TestCase):
    def geometry(self,points=()):
        c,packet,state,contract=public_controller(points=points)
        sub=RaySubcells(contract['bounds'])
        # Public, synthetic empty occupancy isolates the measured-hit branch.
        state.occupancy.occupied_mask[:]=False
        geo=ScanRefinedGeometry(state,c.guard.evidence,c.guard.options,sub)
        return c.guard,geo

    def test_original_answer_and_probe_are_identical_for_pass_and_rejection(self):
        for points in ((),([3.5,3.,4.],)):
            guard,geo=self.geometry(points)
            spec=guard.specs[0];p=np.array([3.,3.,4.]);v=np.array([.8,0,0]);u=np.array([1.,0,0])
            expected=guard.check(p,v,u,spec,geo,[0,0])
            actual=guard.check(p,v,u,spec,ProbeGeometry(geo),[0,0])
            self.assertEqual(expected,actual)

    def test_scan_hit_veto_has_real_offending_points(self):
        guard,geo=self.geometry(([3.5,3.,4.],))
        proxy=ProbeGeometry(geo)
        self.assertFalse(proxy.segment([3,3,4],[3.1,3,4]))
        self.assertEqual(proxy.failure['cause'],'scan_hit_padding')
        self.assertGreater(proxy.failure['offending_hit_count'],0)

    def test_blocked_subcell_is_distinct_from_hit_distance(self):
        guard,geo=self.geometry();geo.subcells.blocked.add((12,12,16))
        proxy=ProbeGeometry(geo);self.assertFalse(proxy.segment([3.01,3.01,4.01],[3.1,3.1,4.1]))
        self.assertEqual(proxy.failure['cause'],'blocked_subcell')

    def test_unknown_piece_coarse_veto_is_reported(self):
        c,packet,state,contract=public_controller()
        state.occupancy.occupied_mask[:]=False
        centers=np.asarray(state.grid.cell_centers)
        index=int(np.argmin(np.linalg.norm(centers-[3,3,4.5],axis=1)))
        state.occupancy.occupied_mask[index]=True
        geo=ScanRefinedGeometry(state,c.guard.evidence,c.guard.options,RaySubcells(contract['bounds']))
        proxy=ProbeGeometry(geo);self.assertFalse(proxy.segment(centers[index],centers[index]+[.01,0,0]))
        self.assertEqual(proxy.failure['cause'],'unobserved_coarse_geometry')

    def test_bounds_are_reported_without_truth_or_rng(self):
        guard,geo=self.geometry();before=np.random.get_state();proxy=ProbeGeometry(geo)
        self.assertFalse(proxy.segment([-1,3,4],[3,3,4]));self.assertEqual(proxy.failure['cause'],'bounds')
        after=np.random.get_state();self.assertTrue(all(np.array_equal(x,y) for x,y in zip(before,after)))

    def test_diagnostic_error_never_changes_original_verdict(self):
        geo=NS(segment=lambda *a,**k:False)
        proxy=ProbeGeometry(geo);self.assertFalse(proxy.segment([0]*3,[1]*3))
        self.assertEqual(proxy.failure['cause'],'REVIEW_REQUIRED')

    def test_observer_preserves_tuple_and_restores_method(self):
        guard,geo=self.geometry(([3.5,3,4],));original=guard.check
        runtime=NS(guard=guard,step=2);o=FailureObserver(runtime)
        args=([3,3,4],[.8,0,0],[1,0,0],guard.specs[0],geo,[0,0])
        self.assertEqual(original(*args),guard.check(*args));self.assertTrue(o.pending)
        o.close();self.assertEqual(guard.check,original)

    def test_old_history_is_bounded_and_frozen_rows_not_reported(self):
        answer=(False,dict(reason='geometry'))
        guard=NS(check=lambda *a:answer);runtime=NS(guard=guard,step=0)
        o=FailureObserver(runtime)
        for step in range(5):
            runtime.step=step;o.check([0]*3,[0]*3,[0]*3,{'a_xy_max':1.3},NS(),[0,0])
        self.assertTrue(all(k[0]>=3 for k in o.pending))
        o.collect([dict(phase=3,emergency=True)]);self.assertFalse(o.records)


if __name__=='__main__':unittest.main()
