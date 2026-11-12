"""Offline scene construction regression; no mission environment or learning."""
import unittest
from unittest.mock import patch
import numpy as np
from core.scenarios.ch3_generator_impl import (
    _unknown_trajectory_record, _validate_unknown_truth_reachability)
from tools.path_option_experiment import special_manifest, CATEGORIES


class SpecialScenarioContracts(unittest.TestCase):
    def test_original_diagonal_trajectory_is_rejected(self):
        agents = np.array([[3., 4., 2.], [3., 8., 4.], [3., 12., 6.], [3., 16., 3.]])
        boxes = [dict(center=[10., 10., 4.], size=[1.2, 12., 7.8])]
        _, _, points = _unknown_trajectory_record([17., 18., 5.], [.30, -.12, .03], boxes)
        with self.assertRaisesRegex(RuntimeError, 'target trajectory enters an oracle-unreachable component'):
            _validate_unknown_truth_reachability(agents, [17., 3., 3.], [17., 18., 5.], points, boxes)

    def test_all_five_layouts_keep_full_trajectory_gate_and_speed(self):
        manifest, faults, labels = special_manifest(count=1)
        self.assertEqual(manifest['scenario_count'], 5)
        self.assertEqual({v['category'] for v in labels.values()}, set(CATEGORIES))
        self.assertEqual(set(labels), set(faults))
        for scene in manifest['scenarios']:
            self.assertTrue(scene['target_trajectory_reachable'])
            self.assertTrue(scene['target_trajectory_exact_endpoint_reachable'])
            self.assertEqual(scene['target_trajectory_connector_failure_count'], 0)
            self.assertEqual(scene['obstacle_family'], 'path_option_challenge_v2')
            self.assertEqual(scene['target_motion_mode'], 'constant_velocity_reflect_v1')
            np.testing.assert_allclose(np.linalg.norm(scene['target_initial_velocity']),
                                       np.linalg.norm([.30, -.12, .03]), rtol=1e-14)
            _, distance, points = _unknown_trajectory_record(scene['target_initial_position'],
                scene['target_initial_velocity'], scene['obstacles'])
            self.assertEqual(len(points), 401)
            self.assertGreater(distance, 10.)
            np.testing.assert_allclose(points[:, 1], 18., rtol=0, atol=0)
            self.assertEqual(faults[scene['scenario_id']][0]['end'], 400)

    def test_feasibility_errors_are_not_swallowed_or_resampled(self):
        with patch('tools.path_option_experiment.generate', return_value={'scenarios': [{}]}), \
             patch('core.scenarios.ch3_generator_impl._validate_unknown_truth_reachability',
                   side_effect=RuntimeError('sentinel feasibility failure')) as gate:
            with self.assertRaisesRegex(RuntimeError, 'sentinel feasibility failure'):
                special_manifest(count=1)
            self.assertEqual(gate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
