import copy
from dataclasses import FrozenInstanceError, fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.launch_assignment import DepartureBoat, LaunchAssignmentMission, first_feasible
from chapter3_bser.local_ctde.navigation import LocalNavigator
from chapter3_bser.neighbor_ctde.exchange import Report
from chapter3_bser.neighbor_ctde.mission import NeighborBoat
from tools.check_launch_candidates import CONFIG, ROOT, SCENE, generators, make_actor, read, write_receipt


def rng(seed):
    return torch.Generator().manual_seed(seed)


class NearbyProposer:
    """Different continuous endpoints close to the same fallback cell."""
    def propose(self, features, generator):
        offset = .000001 * generator.initial_seed()
        unit = np.array([.031 + offset, .033, .037], dtype=np.float32)
        return np.arctanh(unit), unit


class CountingSelector:
    def __init__(self):
        self.calls = 0

    def __call__(self, menu):
        self.calls += 1
        return first_feasible(menu)


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.study = read(CONFIG)
        inputs = ROOT / 'configs/r123_expansion_v1'
        config, options, scenes = [read(inputs / (name + '.json')) for name in ('config', 'options', 'scenes')]
        scene = next(s for s in scenes['scenarios'] if s['scenario_id'] == SCENE)
        cls.mission = LaunchAssignmentMission(config, options, scene, radius=10.,
                                              originals=[rng(i) for i in range(3)],
                                              extras=[rng(i + 7) for i in range(3)], selector_factory=CountingSelector)
        cls.addClassCleanup(cls.mission.close)
        cls.actor = make_actor(config['rl']['hidden_dim'], cls.study['candidates']['proposer_reference'], False)

    def local(self, i=0):
        source = self.mission.boats[i]
        boat = NeighborBoat(copy.deepcopy(source.navigator))
        boat.evidence = copy.deepcopy(source.evidence)
        boat.inbox = copy.deepcopy(source.inbox)
        return boat

    def boat(self, *, extra=True, choose=first_feasible, i=0):
        return DepartureBoat(self.local(i), rng(11), rng(23) if extra else None, choose)

    @staticmethod
    def feasibility(values):
        responses = iter(values)
        def query(nav, point):
            nav.proposal_queries += 1
            return next(responses)
        return patch.object(LocalNavigator, '_can_plan_to', query)

    def test_preview_is_idempotent_readonly_and_reference_matches_native(self):
        boat, old = self.boat(), self.local()
        nav = boat.navigator
        before = (nav.state, nav.env.goal, nav.env.unwrapped.map_module.map_revision, nav._last_request)
        prior_rng = torch.get_rng_state()
        menu = boat.preview(self.actor)
        draws = [g.get_state() for g in (boat.original, boat.extra)]
        queries = boat.proposal_queries
        self.assertIs(menu, boat.preview(self.actor))
        self.assertEqual(queries, boat.proposal_queries)
        self.assertGreater(queries, 0)
        for g, prior in zip((boat.original, boat.extra), draws):
            self.assertTrue(torch.equal(g.get_state(), prior))
        self.assertTrue(torch.equal(prior_rng, torch.get_rng_state()))
        self.assertIs(nav.state, before[0])
        self.assertEqual((nav.env.goal, nav.env.unwrapped.map_module.map_revision, nav._last_request), before[1:])
        self.assertIsNone(nav.point)
        self.assertIsNone(nav.planning_views.filtered_graph)
        self.assertEqual((nav.lifecycle.tasks, nav.manager.current, nav.route_events), ({}, {}, []))
        self.assertEqual(nav.proposal_queries, 0)
        self.assertEqual(self.mission.core.get_task_state().step, 0)
        self.assertEqual(len(menu.points), 2)
        old_command, old_assignment, old_draw = old.act(self.actor, rng(11))
        command, assignment, draw = boat.submit(menu, 0)
        np.testing.assert_array_equal(command, old_command)
        self.assertEqual(assignment, old_assignment)
        for key, value in old_draw.items():
            self.assertEqual(draw[key], value, key)
        self.assertEqual(boat.proposal_queries, queries)
        self.assertEqual(len([e for e in nav.lifecycle.events if e['event'] == 'TASK_ACCEPT']), 1)
        with self.assertRaisesRegex(ValueError, 'consumed'):
            boat.submit(menu, 0)

    def test_local_selector_can_install_the_alternative(self):
        seen = []
        def choose(menu):
            seen.append(menu)
            return np.int64(1)
        boat = self.boat(choose=choose)
        _, assignment, row = boat.act(self.actor, rng(5))
        menu, = seen
        self.assertEqual([field.name for field in fields(menu)], ['boat', 'step', 'features', 'points'])
        self.assertEqual(len(menu.features), 28 + 3 * 15 + 8)
        self.assertFalse(any(menu.features[28:]))
        with self.assertRaises(FrozenInstanceError):
            menu.step = 1
        self.assertEqual(tuple(row['proposed']), menu.points[1])
        self.assertNotEqual(menu.points[0], menu.points[1])
        self.assertEqual(tuple(row['root']), menu.points[1])
        self.assertEqual(assignment.planned_path, row['departure']['route_points'])
        self.assertEqual(json.loads(json.dumps(row))['departure']['selected_option'], 1)
        self.assertTrue(row['accepted'])
        self.assertTrue(row['departure']['route_to_selected'])
        self.assertEqual(row['departure']['selected_stream'], 'extra_offer')
        self.assertEqual((row['departure']['task_id'], boat.lifecycle.serial), (1, 1))

    def test_distinct_raw_draws_same_projected_cell_are_one_option(self):
        boat = self.boat()
        with self.feasibility([False, True, False, True]):
            menu = boat.preview(NearbyProposer())
        self.assertEqual(len(menu.points), 1)
        _, _, row = boat.submit(menu, 0)
        a, b = row['departure']['draws']
        self.assertNotEqual(a['raw_goal'], b['raw_goal'])
        self.assertEqual(a['projected'], b['projected'])
        self.assertEqual((a['option'], b['option'], b['status']), (0, 0, 'duplicate'))
        self.assertEqual((a['queries'], b['queries'], boat.proposal_queries), (2, 2, 4))

    def test_failed_reference_keeps_extra_as_first_effective_option(self):
        boat = self.boat()
        with self.feasibility([False, False, True]):
            _, _, row = boat.act(self.actor, rng(51))
        d = row['departure']
        self.assertEqual((d['selected_option'], d['selected_stream']), (0, 'extra_offer'))
        self.assertEqual([item['status'] for item in d['draws']], ['infeasible', 'feasible'])
        self.assertEqual([item['queries'] for item in d['draws']], [2, 1])
        self.assertEqual(boat.proposal_queries, 3)

    def test_empty_menu_enters_native_wait_and_never_installs_a_search_root(self):
        boat = self.boat(choose=lambda menu: self.fail('empty menu called selector'))
        with self.feasibility([False] * 4):
            command, item, row = boat.act(self.actor, rng(7))
        self.assertIsNone(row['proposed'])
        self.assertFalse(row['accepted'])
        d = row['departure']
        self.assertEqual((d['options'], d['selected_option'], d['task_id'], d['preview_queries']), ((), None, 0, 4))
        self.assertEqual(d['fallback'], 'no_feasible_offer')
        self.assertFalse(d['route_to_selected'])
        self.assertEqual(boat.lifecycle.serial, 0)
        self.assertTrue(boat.lifecycle.tasks[0].released)
        self.assertTrue(np.isfinite(command).all())
        self.assertEqual(item.assignment_id, '0')

    def test_candidate_projection_can_be_rejected_at_native_admission(self):
        boat = self.boat()
        menu = boat.preview(self.actor)
        with patch.object(boat.navigator, 'assignment', return_value=dict(assigned=False, token=None, point=None)):
            _, _, row = boat.submit(menu, 0)
        self.assertIsNotNone(row['proposed'])
        self.assertFalse(row['accepted'])
        self.assertEqual(row['departure']['fallback'], 'native_rejection')
        self.assertFalse(row['departure']['route_to_selected'])
        self.assertEqual(row['departure']['selected_stream'], 'original_offer')

    def test_foreign_stale_or_invalid_menu_cannot_consume_the_request(self):
        boat, peer = self.boat(), self.boat(i=1)
        menu, foreign = boat.preview(self.actor), peer.preview(self.actor)
        for bad in (foreign, replace(menu), None):
            with self.assertRaisesRegex(ValueError, 'foreign'):
                boat.submit(bad, 0)
        for choice in (-1, None, True, 2):
            with self.assertRaisesRegex(ValueError, 'feasible'):
                boat.submit(menu, choice)
        boat.navigator.features[0] += .25
        with self.assertRaisesRegex(ValueError, 'observation changed'):
            boat.submit(menu, 0)
        self.assertEqual(boat.lifecycle.serial, 0)

    def test_withheld_peer_information_cannot_affect_offers_or_admission(self):
        first, second = self.boat(), self.boat()
        a = first.preview(self.actor)
        peer = self.mission.boats[1]
        real = self.mission.core.unwrapped
        with patch.object(peer.navigator, 'point', np.array([8., 4., 2.])), \
             patch.object(peer.navigator, 'features', np.ones_like(peer.features)), \
             patch.object(real, '_agent_pos', real._agent_pos + 17.):
            b = second.preview(self.actor)
        self.assertEqual(a, b)
        cmd_a, item_a, row_a = first.submit(a, 1)
        cmd_b, item_b, row_b = second.submit(b, 1)
        np.testing.assert_array_equal(cmd_a, cmd_b)
        self.assertEqual((item_a, row_a), (item_b, row_b))
        late = self.boat()
        late.inbox.add(Report(1, 0, (1., 2., 3.), (0.,) * 3, 1, True, (4.,) * 3, (), (), False))
        with self.assertRaisesRegex(ValueError, 'undelivered'):
            late.preview(self.actor)
        self.assertEqual(late.proposal_queries, 0)

    def test_failed_preview_retry_keeps_streams_and_charges_real_queries(self):
        boat, clean = self.boat(), self.boat()
        def broken(probe, point):
            probe.proposal_queries += 1
            raise RuntimeError('projection interrupted')
        with patch.object(LocalNavigator, '_can_plan_to', broken), self.assertRaisesRegex(RuntimeError, 'interrupted'):
            boat.preview(self.actor)
        for rng_a, rng_b in ((boat.original, clean.original), (boat.extra, clean.extra)):
            self.assertTrue(torch.equal(rng_a.get_state(), rng_b.get_state()))
        self.assertEqual(boat.preview(self.actor), clean.preview(self.actor))
        self.assertEqual(boat.proposal_queries, clean.proposal_queries + 1)

    def test_zero_query_invalid_start_and_interrupted_native_admission(self):
        boat = self.boat()
        with patch('chapter3_bser.local_ctde.navigation.start_status', return_value='invalid_start'):
            menu = boat.preview(self.actor)
        self.assertEqual((menu.points, boat.proposal_queries), ((), 0))
        with patch.object(boat.navigator, 'act', side_effect=RuntimeError('abort')):
            with self.assertRaisesRegex(RuntimeError, 'abort'):
                boat.submit(menu, None)
        with self.assertRaisesRegex(RuntimeError, 'fresh mission'):
            boat.act(self.actor)
        self.assertEqual(boat.lifecycle.serial, 0)

    def test_selector_instances_and_executor_are_isolated_at_installation(self):
        boats = self.mission.boats
        menu = self.boat().preview(self.actor)
        self.assertEqual(boats[0].choose(menu), 0)
        self.assertEqual([b.choose.calls for b in boats[:3]], [1, 0, 0])
        self.assertIs(type(boats[3]), NeighborBoat)
        with self.assertRaisesRegex(ValueError, 'only search'):
            DepartureBoat(boats[3], rng(2), None, first_feasible)
        with self.assertRaisesRegex(ValueError, 'separate'):
            LaunchAssignmentMission(None, None, None, radius=10., originals=[rng(i) for i in range(3)],
                                    selector_factory=lambda: boats[0].choose)

    def test_continuation_is_separate_and_executor_keeps_native_standby(self):
        streams = self.study['random_streams']
        family = 'fixture'
        one = self.boat(extra=False)
        two = self.boat()
        cont_a, cont_b = (generators(streams, family, 'continuation')[0] for _ in range(2))
        one.act(self.actor, cont_a)
        two.act(self.actor, cont_b)
        self.assertTrue(torch.equal(cont_a.get_state(), cont_b.get_state()))
        self.assertTrue(torch.equal(one.original.get_state(), two.original.get_state()))
        # Deliberately open a later request on the isolated native instances.
        for boat in (one, two):
            boat.navigator.env.frame = replace(boat.env.frame, step=10)
            boat.navigator._release_requests.add(0)
        cmd_a, item_a, draw_a = one.act(self.actor, cont_a)
        cmd_b, item_b, draw_b = two.act(self.actor, cont_b)
        np.testing.assert_array_equal(cmd_a, cmd_b)
        self.assertEqual((item_a, draw_a), (item_b, draw_b))
        self.assertEqual(draw_a['step'], 10)
        self.assertNotIn('departure', draw_a)
        self.assertFalse(torch.equal(cont_a.get_state(), generators(streams, family, 'continuation')[0].get_state()))
        executor = self.local(3)
        _, assignment, draw = executor.act(None)
        self.assertIsNone(draw)
        self.assertEqual(assignment.final_waypoint, tuple(executor.point))

    def test_aliases_and_legacy_training_are_rejected_before_rollout(self):
        originals = [rng(i) for i in range(3)]
        with self.assertRaisesRegex(ValueError, 'independent'):
            LaunchAssignmentMission(None, None, None, radius=10., originals=originals, extras=originals)
        with self.assertRaisesRegex(ValueError, 'private'):
            LaunchAssignmentMission(None, None, None, radius=10., originals=[torch.default_generator, *originals[:2]])
        mission = LaunchAssignmentMission.__new__(LaunchAssignmentMission)
        mission._offer_streams = tuple(originals)
        with self.assertRaisesRegex(ValueError, 'independent'):
            mission.rollout(None, originals, 1)
        with self.assertRaisesRegex(ValueError, 'on-policy'):
            mission.rollout(None, originals, 1, log_training=True)

    def test_receipt_is_complete_or_absent_and_cannot_replace_prior_evidence(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'replay.json'
            with self.assertRaises(TypeError):
                write_receipt(path, {'invalid': np.float32(1.)})
            self.assertFalse(path.exists())
            write_receipt(path, {'options': [0, 1]})
            with self.assertRaises(FileExistsError):
                write_receipt(path, {})
            self.assertEqual(read(path), {'options': [0, 1]})


if __name__ == '__main__':
    unittest.main()
