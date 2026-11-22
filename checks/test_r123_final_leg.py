"""Final-leg gates, reversible cursor, and action-level safety contracts (no env)."""
import unittest
import copy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from chapter3_bser.path_option.path_context import Route
from chapter3_bser.r123.options import CONTROL
from chapter3_bser.r123_final_leg import FinalLegManager, RuleRuntime, connection_verdict
from chapter3_bser.r123_repair.lifecycle import Lifecycle
from chapter3_bser.r123_task_sync.diagnostics import ObservedRuntime
from chapter3_bser.r123_tracking.control import conditional_banks
from chapter3_bser.r123_tracking.progress import tracking_velocity
from chapter3_bser.reach_avoid.options import CONTRACT, SPECS
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.options import validate
from checks.test_r123_governed import geometry, packet


class FinalLegContracts(unittest.TestCase):
    def setUp(self):
        self.position = np.array([7.92, 1., 6.5])
        self.endpoint = np.array([8.28, 1.16, 5.63])
        self.route = Route([self.position, [7., 1., 6.5], [9., 1., 6.5], self.endpoint],
                           revision=287, query_step=336, reachable=True, index=1)
        lifecycle = Lifecycle(CONTROL)
        lifecycle.accept(3, self.endpoint, 2, 336, self.position)
        self.manager = FinalLegManager(dict(stagnation_steps=30), lifecycle, True)
        self.manager.public_pose[3] = (True, 3.)
        self.c = self.manager.commit(3, self.endpoint, self.endpoint, self.endpoint, 336, 2,
                                     self.route, reason='test')
        self.geo = geometry()
        self.geo.spacing = np.array([2., 2., 1.])
        self.geo.raw = SimpleNamespace(segment_free=Mock(return_value=True))
        self.geo.segment = Mock(return_value=True)
        self.manager.geometry = self.geo

    def status(self, step=336, **overrides):
        args = dict(agent=3, position=self.position, velocity=np.zeros(3), phase=2,
                    step=step, endpoint_free=True, target_velocity=np.zeros(3))
        return self.manager.status(**dict(args, **overrides))

    def test_shortcut_scopes_feedforward_to_existing_committed_endpoint(self):
        p = packet()
        p.phases[3] = 2
        p.target_velocity = np.array([.1, .2, -.3])
        route_id = self.route.route_id
        np.testing.assert_equal(tracking_velocity(p, 3, self.route, self.endpoint, True), np.zeros(3))
        self.status()
        self.assertEqual(self.manager.verdict, 'direct')
        self.assertEqual(self.route.route_id, route_id)
        np.testing.assert_allclose(self.route.tracking(self.position), self.endpoint, atol=1e-6)
        np.testing.assert_equal(tracking_velocity(p, 3, self.route, self.endpoint, True), p.target_velocity)
        np.testing.assert_equal(self.manager.lifecycle.tasks[3].last_tracking, self.route.points[-1])

    def test_new_scan_veto_reinstates_original_uncompleted_vertex(self):
        self.status()
        self.assertEqual(self.route.index, 3)
        self.geo.segment.return_value = False
        self.status(337)
        self.assertEqual(self.manager.verdict, 'scan_veto')
        self.assertEqual(self.route.index, 1)

    def test_known_occupied_grid_veto_precedes_scan_check(self):
        self.geo.raw.segment_free.return_value = False
        self.status()
        self.assertEqual(self.manager.verdict, 'occupied_grid')
        self.assertEqual(self.route.index, 1)
        self.geo.segment.assert_not_called()

    def test_outside_one_grid_spacing_and_already_terminal_use_original_cursor(self):
        self.assertEqual(connection_verdict(self.c, self.endpoint + [2.01, 0., 0.], self.geo), 'outside_final_leg')
        self.route.index = len(self.route.points) - 1
        self.assertEqual(connection_verdict(self.c, self.position, self.geo), 'already_terminal')
        self.geo.segment.assert_not_called()

    def test_failure_connector_changed_endpoint_and_recovery_keep_grid(self):
        self.route.failure = 'public_ray_start_connector'
        self.assertEqual(connection_verdict(self.c, self.position, self.geo), 'grid_required')
        self.route.failure = ''
        self.route.reachable = False
        self.assertEqual(connection_verdict(self.c, self.position, self.geo), 'grid_required')
        self.route.reachable = True
        self.c.endpoint = self.endpoint + .15
        self.assertEqual(connection_verdict(self.c, self.position, self.geo), 'different_endpoint')
        self.c.mode = 'RECOVER'
        self.assertEqual(connection_verdict(self.c, self.position, self.geo), 'grid_required')

    def test_no_selection_for_released_invalid_standby_or_frozen_task(self):
        for values in (dict(endpoint_free=False), dict(phase=1), dict(phase=3)):
            self.status(**values)
            self.assertEqual(self.route.index, 1)
        self.manager.lifecycle.release(3, 336, 'UNASSIGNED')
        self.status()
        self.assertEqual(self.route.index, 1)
        self.geo.segment.assert_not_called()

    def test_query_never_inherits_shortcut_as_completed_grid_progress(self):
        self.status()
        runtime = RuleRuntime.__new__(RuleRuntime)
        runtime.direct_final_leg = True
        runtime.manager = self.manager
        with patch.object(ObservedRuntime, '_query', side_effect=lambda *_: self.route.index):
            self.assertEqual(runtime._query(3, self.endpoint, object()), 1)
        self.assertIsNone(self.manager.saved_cursor)

    def test_search_and_disabled_options_are_parent_options(self):
        runtime = RuleRuntime.__new__(RuleRuntime)
        with patch.object(ObservedRuntime, '_options', return_value=object()) as parent:
            for runtime.direct_final_leg, agent in ((True, 0), (True, 2), (False, 3)):
                self.assertIs(runtime._options(agent, None, None), parent.return_value)
        self.assertEqual(parent.call_count, 3)

    def test_direct_segment_still_uses_braking_backup_and_speed_limits(self):
        self.status()
        self.assertEqual(self.manager.verdict, 'direct')
        p = packet()
        p.phases[3] = 2
        p.positions[3] = self.position
        track = self.route.tracking(self.position)
        guard = SafetyController(SPECS, CONTRACT, validate())
        # Clear nominal segment; the actual guard's robust motion/backup corridors can veto it.
        self.geo.segment.side_effect = lambda *args, **kwargs: False
        with patch.object(guard, '_caps', return_value=(.2, .1, 1.)):
            _, mask, forced, emergency, diag = conditional_banks(
                p, 3, [track], self.geo, guard, target_velocities=[np.zeros(3)])
        self.assertFalse(mask[0, :7].any())
        self.assertTrue(emergency[0])
        self.assertEqual(forced[0], 8)
        self.assertIn('motion_or_backup_geometry', [d['reason'] for d in diag[0]['candidate_checks']])


class PairingContracts(unittest.TestCase):
    def setUp(self):
        item = dict(step=0, agent=0, phase=0, mode='TASK', route_id='a', route_index=1,
                    reason='CONTINUE', emergency=False, decision={'action': 0})
        for k in ('position', 'next_position', 'velocity', 'executed', 'event_target', 'endpoint'):
            item[k] = [0., 0., 0.]
        self.grid = dict(actual_length=1, found_step=None, executor_target_received_step=None,
                         path_option={'trace': [dict(item, agent=i) for i in range(4)]})
        self.direct = copy.deepcopy(self.grid)
        self.direct['final_leg'] = {'enabled': True}

    def test_no_activation_compares_the_entire_episode(self):
        from tools.r123_final_leg import compare
        self.assertEqual(compare(self.grid, self.direct, dict(first_direct_action=None)), 1)
        self.direct['path_option']['trace'][2]['next_position'] = [.1, 0., 0.]
        with self.assertRaisesRegex(ValueError, 'physics'):
            compare(self.grid, self.direct, dict(first_direct_action=None))

    def test_wrong_arm_and_changed_handoff_cannot_pass_as_pairs(self):
        from tools.r123_final_leg import compare
        self.direct['final_leg']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'candidate'):
            compare(self.grid, self.direct, dict(first_direct_action=0))
        self.direct['final_leg']['enabled'] = True
        self.direct['executor_target_received_step'] = 2
        with self.assertRaisesRegex(ValueError, 'handoff'):
            compare(self.grid, self.direct, dict(first_direct_action=0))


if __name__ == '__main__':
    unittest.main()
