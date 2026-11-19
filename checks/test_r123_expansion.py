"""Contract and raw-outcome tests; no native environment construction."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.r123_expansion import contract, jobs, wilson, audit_episode, run_one, DATA, read


def audit_fixture(capture=False):
    steps = 7 if capture else 2
    scene = dict(max_steps=steps, target_initial_position=[1., 1., 1.],
                 target_initial_velocity=[.3, 0., 0.], target_motion_mode='constant_velocity_reflect_v1', obstacles=[])
    trace = []
    for t in range(steps):
        for a in range(4):
            searching = a < 3 and (not capture or t == 0)
            position = [18., 18., 6.]
            next_position = position.copy()
            phase = 0 if searching else 3 if a < 3 else 1
            if capture and (a == 0 and t == 0 or a == 3 and t >= 2):
                position = [1. + .06 * t, 1., 1.]
                next_position = [1. + .06 * (t + 1), 1., 1.]
                if a == 3:
                    phase = 2
            trace.append(dict(step=t, agent=a, phase=phase, position=position, next_position=next_position,
                velocity=[0., 0., 0.], endpoint=[19., 19., 7.], decision={'action': 0}, reason='CONTINUE'))
    raw = dict(actual_length=steps, task_network_trained=False, path_option={'trace': trace},
        found_step=1 if capture else None, executor_target_received_step=2 if capture else None,
        capture_contact_step_count=5 if capture else 0, capture_hold_counter_max=5 if capture else 0, success=capture)
    return scene, raw


class ExpansionChecks(unittest.TestCase):
    def dataset(self):
        return [read(DATA / n) for n in ('config.json', 'options.json', 'scenes.json', 'exclusions.json')]

    def test_24_new_physical_scenes_have_original_contract(self):
        c, o, s, x = self.dataset()
        self.assertEqual(len(contract(c, o, s, x)), 24)
        self.assertEqual(len({v['scenario_seed'] for v in s['scenarios']}), 24)

    def test_renamed_physical_duplicate_is_rejected(self):
        c, o, s, x = self.dataset()
        identity = s['scenarios'][1]['scenario_id']
        seed = s['scenarios'][1]['scenario_seed']
        s['scenarios'][1] = copy.deepcopy(s['scenarios'][0])
        s['scenarios'][1].update(scenario_id=identity, scenario_seed=seed)
        with self.assertRaises(ValueError):
            contract(c, o, s, x)

    def test_overlap_with_old_inputs_is_rejected(self):
        from chapter3_bser.learning.config import scenario_fingerprint
        c, o, s, x = self.dataset()
        x['fingerprints'].append(scenario_fingerprint(s['scenarios'][0]))
        with self.assertRaises(ValueError):
            contract(c, o, s, x)

    def test_scene_count_and_step_budget_cannot_expand_silently(self):
        c, o, s, x = self.dataset()
        s['scenarios'].pop()
        with self.assertRaises(ValueError):
            contract(c, o, s, x)
        c, o, s, x = self.dataset()
        c['max_steps'] = 401
        with self.assertRaises(ValueError):
            contract(c, o, s, x)

    def test_jobs_pair_same_seed_and_alternate_execution_order(self):
        planned = jobs()
        self.assertEqual(len(planned), 48)
        self.assertEqual(len({v['folder'] for v in planned}), 48)
        for n in range(24):
            pair = planned[n * 2:n * 2 + 2]
            self.assertEqual({j['scene'] for j in pair}, {n})
            self.assertEqual(len({j['seed'] for j in pair}), 1)
            self.assertEqual({j['arm'] for j in pair}, {'tracking_both', 'task_sync'})
        self.assertNotEqual(planned[0]['arm'], planned[2]['arm'])

    def test_wilson_endpoints_symmetry_and_invalid_count(self):
        self.assertAlmostEqual(wilson(0, 24)[0], 0.)
        self.assertAlmostEqual(wilson(24, 24)[1], 1.)
        self.assertAlmostEqual(sum(wilson(12, 24)), 1.)
        with self.assertRaises(ValueError):
            wilson(25, 24)

    def test_raw_detection_and_original_five_interval_hold(self):
        scene, raw = audit_fixture(True)
        curve = audit_episode(raw, scene)
        self.assertEqual(int(curve[1, 0]), 3)
        raw['capture_hold_counter_max'] = 4
        with self.assertRaises(ValueError):
            audit_episode(raw, scene)

    def test_missing_agent_or_false_discovery_is_rejected(self):
        scene, raw = audit_fixture(False)
        audit_episode(raw, scene)
        raw['found_step'] = 1
        with self.assertRaises(ValueError):
            audit_episode(raw, scene)
        scene, raw = audit_fixture(False)
        raw['path_option']['trace'].pop()
        with self.assertRaises(ValueError):
            audit_episode(raw, scene)

    def test_smoke_bound_rejects_more_than32_before_initialization(self):
        c, o, s, x = self.dataset()
        c['max_steps'] = 33
        s['scenarios'][0]['max_steps'] = 33
        ledger = dict(environment_initializations=0, physical_steps=0)
        with self.assertRaises(ValueError):
            run_one(c, s['scenarios'][0], o, 0, 'task_sync', ledger=ledger, smoke=True)
        self.assertEqual(ledger['environment_initializations'], 0)


if __name__ == '__main__':
    unittest.main()
