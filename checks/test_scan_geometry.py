"""Local collision-model regressions; no environment initialization or steps."""
import copy
import unittest
from types import SimpleNamespace as NS
import numpy as np
from checks.test_safe_rl import state
from chapter3_bser.safe_rl.evidence import EvidenceGeometry, ScanEvidence
from chapter3_bser.safe_rl.options import validate
from chapter3_bser.reach_avoid.scan_geometry import RaySubcells, ScanRefinedGeometry, segment_pieces, ray_cells
from core.env.task_protocol import closed_segment_aabb_first_hit


def event(origin=(3.1,3.1,4.1), direction=(1.,0.,0.), distance=.7, hit=True, step=0, sequence=1):
    return NS(sequence=sequence, physical_step=step, kind='scan', payload=NS(
        origins=np.array([origin]), directions=np.array([direction]),
        distances=np.array([[distance]]), hit_mask=np.array([[hit]])))


def geometry(events=None):
    s = state(occupied=[np.ravel_multi_index((1,1,4),(10,10,8))])
    evidence, subcells = ScanEvidence(), RaySubcells((20.,20.,8.))
    events = [event()] if events is None else events
    evidence.consume(events)
    subcells.consume(events, 10)
    return ScanRefinedGeometry(s,evidence,validate(),subcells), s, evidence


class ScanGeometry(unittest.TestCase):
    def test_same_coarse_cell_is_not_physical_collision(self):
        g, _, _ = geometry()
        p, q = [3.1,3.1,4.1], [3.15,3.1,4.1]
        self.assertFalse(g.raw.segment_free(p,q))
        self.assertTrue(g.segment(p,q))
        self.assertIsNone(closed_segment_aabb_first_hit(p,q,[3.75,3.,4.],[4.,3.3,4.5]))

    def test_true_contact_and_tunneling_still_count(self):
        lo, hi = [3.75,3.,4.], [4.,3.3,4.5]
        for end in ([3.75,3.1,4.1], [4.5,3.1,4.1]):
            self.assertIsNotNone(closed_segment_aabb_first_hit([3.1,3.1,4.1],end,lo,hi))

    def test_ray_hit_padding_not_removed_by_free_subcells(self):
        g, _, _ = geometry()
        self.assertFalse(g.segment([3.1,3.1,4.1],[3.6,3.1,4.1]))

    def test_unobserved_part_of_same_coarse_cell_stays_blocked(self):
        g, _, _ = geometry()
        self.assertFalse(g.segment([3.1,3.1,4.1],[3.1,3.5,4.1]))

    def test_no_scan_evidence_cannot_clear_current_cell(self):
        g, _, _ = geometry([])
        self.assertEqual(g.distance([3.1,3.1,4.1]),0.)
        self.assertFalse(g.segment([3.1,3.1,4.1],[3.15,3.1,4.1]))

    def test_parent_geometry_and_map_are_unchanged(self):
        g,s,e = geometry(); before=copy.deepcopy(s.occupancy.occupied_mask)
        self.assertFalse(EvidenceGeometry(s,e,validate()).segment([3.1,3.1,4.1],[3.15,3.1,4.1]))
        np.testing.assert_array_equal(before,s.occupancy.occupied_mask)
        self.assertGreater(g.distance([3.1,3.1,4.1]),.52)

    def test_hit_wins_simultaneous_and_later_free_ray(self):
        hit = event(distance=.4)
        free = event(distance=1.,hit=False,step=1,sequence=2)
        g,_,_=geometry([hit,free])
        self.assertFalse(g.subcells.contains([[3.5,3.1,4.1]])[0])
        self.assertFalse(g.observed_path([3.1,3.1,4.1],[3.8,3.1,4.1]))

    def test_endpoint_free_does_not_allow_unobserved_middle(self):
        g,_,_=geometry([event(distance=.05,hit=False),event(origin=(3.9,3.1,4.1),distance=.05,hit=False,sequence=2)])
        self.assertFalse(g.observed_path([3.1,3.1,4.1],[3.9,3.1,4.1]))

    def test_subcell_split_visits_thin_crossings(self):
        _,_,keys=segment_pieces([.24,.249,1.1],[.76,.251,1.1])
        self.assertIn((2,1,4),map(tuple,keys))

    def test_free_ray_on_subcell_faces_visits_both_sides(self):
        cells=ray_cells([3.999,3.,4.5],[3.,3.,4.5])
        self.assertIn((14,11,17),cells)
        self.assertIn((14,12,18),cells)

    def test_original_near_buffer_public_observation_preserves_all_four_exits(self):
        from tools.check_scan_geometry import buffer_initial_checks
        r=buffer_initial_checks()
        self.assertTrue(r['passed']);self.assertEqual(len(r['roles']),4)
        self.assertEqual(r['environment_initializations'],0)

    def test_nonfinite_and_boundary_escape_are_rejected(self):
        g,_,_=geometry()
        for q in ([np.nan,3.,4.],[-.1,3.,4.],[20.1,3.,4.]):
            self.assertFalse(g.segment([3.1,3.1,4.1],q))

    def test_future_and_duplicate_events_rejected(self):
        sub=RaySubcells((20.,20.,8.))
        with self.assertRaises(ValueError):sub.consume([event(step=2)],1)
        sub.consume([event()],0)
        with self.assertRaises(ValueError):sub.consume([event()],0)

    def test_outward_exit_can_leave_initial_mixed_subcell(self):
        from checks.test_safe_rl import SPECS, CONTRACT
        from chapter3_bser.reach_avoid.scan_geometry import refined_buffer_exit
        g,_,_=geometry([event(distance=.1),event(direction=(-1.,0.,0.),distance=2.,hit=False,sequence=2)])
        p=np.array([3.1,3.1,4.1])
        self.assertFalse(g.segment(p,p+[-.02,0.,0.]))
        result=refined_buffer_exit(p,np.zeros(3),SPECS[3],g,CONTRACT,validate(),np.zeros(2))
        self.assertIsNotNone(result)
        self.assertLess(result[0][0],0.)
        self.assertTrue(result[1]['public_subcell_corridor_checked'])

    def test_outward_exit_cannot_clear_unobserved_rest_of_coarse_cell(self):
        from checks.test_safe_rl import SPECS, CONTRACT
        from chapter3_bser.reach_avoid.scan_geometry import refined_buffer_exit
        g,_,_=geometry([event(distance=.1)])
        self.assertIsNone(refined_buffer_exit([3.1,3.1,4.1],np.zeros(3),SPECS[3],g,CONTRACT,validate(),np.zeros(2)))

    def test_frozen_0006_three_roles_have_checked_motion(self):
        from tools.check_scan_geometry import frozen_checks
        r=frozen_checks()
        self.assertTrue(r['passed']);self.assertEqual(len(r['records']),3)
        self.assertEqual(r['environment_initializations'],0)
        for row in r['records']:
            self.assertGreater(row['checked_moving_candidates'],0)
            self.assertTrue(row['selected_model_checked'])

    def test_local_goal_event_uses_replayable_pose_evidence(self):
        from checks.reach_avoid_fixtures import public_controller
        from chapter3_bser.reach_avoid.goals import Goal, CAPTURE, goal_reached
        c,p,_,_=public_controller()
        cell=tuple(np.floor(p.positions[0]/p.spacing).astype(int))
        index=np.ravel_multi_index(cell,tuple(p.shape))
        p.free[index]=False;p.occupied[index]=True;p.clearances[0]=1.
        goal=Goal(tuple(p.positions[0]),tuple(p.velocities[0]),CAPTURE)
        p.pose_free[0]=True
        self.assertTrue(goal_reached(p,0,goal,c.options))
        p.pose_free[0]=False
        self.assertFalse(goal_reached(p,0,goal,c.options))

