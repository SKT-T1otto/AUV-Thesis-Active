"""Public evidence correctness with synthetic payloads; no simulator episodes."""
from dataclasses import replace
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch
import numpy as np
from core.env.public_events import PublicEventJournal, ScanObservation, StepObservation, payload_digest, vectors
from chapter3_bser.experiments.coupled_search_v1.public_observations import PublicObservationAdapter
from chapter3_bser.experiments.coupled_search_v1.audit import installed_route_hash, PublicEvidenceSession
from chapter3_bser.experiments.coupled_search_v1.config import DEFAULT, validate_research_config
from chapter3_bser import planner_factory


def scan():
    directions = np.asarray([v for v in np.ndindex(3, 3, 3) if v != (1, 1, 1)], dtype=float) - 1
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return ScanObservation(vectors(np.zeros((4, 3))), vectors(directions), vectors(np.ones((4, 26))),
                           tuple((False,)*26 for _ in range(4)), (0, 1, 2, 3))


def transition(*, collision=False, found=False, timeout=False):
    p = vectors(np.zeros((4, 3)))
    return StepObservation(p, p, p, p, False, found, not collision, not collision,
                           collision or timeout, 'obstacle_collision' if collision else 'timeout' if timeout else 'running')


def journal(capacity=1024):
    j = PublicEventJournal('example', capacity=capacity)
    j.reset(); j.append(0, 'shared_neighbor26', 'scan', scan())
    return j


def guidance(path=((0., 0., 0.), (1., 0., 0.)), *, hold=False, tracking=(1., 0., 0.)):
    item = SimpleNamespace(agent_id=0, role='search_fast', assignment_id='same_id', final_waypoint=(1., 0., 0.),
                           planned_path=path, hold_state=hold, reachable=not hold, hold_position=(0., 0., 0.),
                           execution_request=False, tracking_waypoint=tracking)
    return SimpleNamespace(agent_assignments=(item,), step=0, decision_reason='example')


class PublicEvents(unittest.TestCase):
    def test_reset_scan_is_not_negative_evidence(self):
        a = PublicObservationAdapter(); rows = a.ingest(journal().read())
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0].physical_step, 0)
        self.assertFalse(rows[0].negative_observation_valid)

    def test_repeated_transaction_is_idempotent_but_next_step_scan_is_new(self):
        j = journal(); a = PublicObservationAdapter(); events = j.read()
        a.ingest(events); self.assertEqual(a.ingest(events), ())
        j.append(1, 'shared_neighbor26', 'scan', scan())
        j.append(1, 'team_search_detector', 'step', transition())
        row, = a.ingest(j.read(a.cursor))
        self.assertEqual(len(row.scans), 1); self.assertTrue(row.negative_observation_valid)

    def test_collision_does_not_publish_negative_evidence(self):
        j = journal(); j.append(1, 'team_search_detector', 'step', transition(collision=True))
        rows = PublicObservationAdapter().ingest(j.read())
        self.assertFalse(rows[-1].negative_observation_valid)
        self.assertTrue(rows[-1].transition.payload.episode_terminated)

    def test_found_and_timeout_validity(self):
        for found in (False, True):
            j = journal(); j.append(1, 'shared_neighbor26', 'scan', scan())
            j.append(1, 'team_search_detector', 'step', transition(found=found, timeout=True))
            row = PublicObservationAdapter().ingest(j.read())[-1]
            self.assertEqual(row.negative_observation_valid, not found)

    def test_incomplete_check_and_post_found_check_rejected(self):
        for payload in (replace(transition(), target_transition_completed=False),
                        replace(transition(found=True), found_before=True)):
            j = journal(); j.append(1, 'team_search_detector', 'step', payload)
            with self.assertRaises(ValueError): PublicObservationAdapter().ingest(j.read())

    def test_conflicting_duplicate_and_hash_tamper_rejected(self):
        j = journal(); event, = j.read(); a = PublicObservationAdapter(); a.ingest((event,))
        payload = replace(event.payload, distances=vectors(np.full((4, 26), 2.)))
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            a.ingest((replace(event, payload=payload, payload_sha256=payload_digest(payload)),))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            a.ingest((replace(event, payload=payload),))

    def test_batch_rejection_is_atomic(self):
        j = journal(); event, = j.read(); a = PublicObservationAdapter()
        invalid = replace(event, sequence=3, physical_step=2)
        with self.assertRaises(ValueError): a.ingest((event, invalid))
        self.assertEqual(a.cursor, 0); self.assertIsNone(a.episode_id)

    def test_missing_steps_and_reversed_sequence_rejected(self):
        j = journal(); event, = j.read(); a = PublicObservationAdapter(); a.ingest((event,))
        with self.assertRaisesRegex(ValueError, 'missing physical'):
            a.ingest((replace(event, physical_step=2, sequence=2),))
        with self.assertRaisesRegex(ValueError, 'sequence gap'):
            a.ingest((replace(event, sequence=3),))

    def test_reset_changes_episode_identity_and_requires_adapter_reset(self):
        j = journal(); a = PublicObservationAdapter(); a.ingest(j.read())
        old = j.episode_id; j.reset(); j.append(0, 'shared_neighbor26', 'scan', scan())
        self.assertNotEqual(old, j.episode_id)
        with self.assertRaisesRegex(ValueError, 'mixed episodes'): a.ingest(j.read())
        a.reset(); self.assertEqual(len(a.ingest(j.read())), 1)

    def test_evicted_history_never_silently_accepted(self):
        j = journal(2); j.append(1, 'shared_neighbor26', 'scan', scan())
        j.append(1, 'team_search_detector', 'step', transition())
        with self.assertRaisesRegex(ValueError, 'evicted'): j.read()
        with self.assertRaises(ValueError): j.read(-1)
        with self.assertRaises(ValueError): j.read(999)

    def test_scan_geometry_and_source_validation(self):
        for payload in (replace(scan(), agent_ids=(0, 1, 2)),
                        replace(scan(), distances=vectors(np.full((4, 26), -1.))),
                        replace(scan(), directions=vectors(np.zeros((26, 3))))):
            j = PublicEventJournal('bad'); j.reset(); j.append(0, 'shared_neighbor26', 'scan', payload)
            with self.assertRaises(ValueError): PublicObservationAdapter().ingest(j.read())

    def test_route_hash_covers_safety_replacement_but_not_tracker_progress(self):
        one = guidance(); other = guidance(path=((0., 0., 0.), (0., 1., 0.), (1., 0., 0.)))
        self.assertNotEqual(installed_route_hash(one), installed_route_hash(other))
        self.assertNotEqual(installed_route_hash(one), installed_route_hash(guidance(hold=True)))
        self.assertEqual(installed_route_hash(one), installed_route_hash(guidance(tracking=(.5, 0., 0.))))

    def test_audit_invalidity_is_visible_and_never_changes_d2(self):
        s = PublicEvidenceSession(DEFAULT, (), {})
        event, = journal().read(); s.consume((replace(event, sequence=2),))
        self.assertFalse(s.summary()['valid']); self.assertFalse(s.summary()['algorithm_changed'])
        self.assertEqual(s.consume((event,)), ())

    def test_disabled_configuration_and_unsupported_algorithm(self):
        self.assertFalse(validate_research_config()['enabled'])
        with self.assertRaises(ValueError): validate_research_config(dict(DEFAULT, mode='coupled_search'))
        with self.assertRaises(ValueError): validate_research_config(dict(DEFAULT, enabled=1))
        owner = SimpleNamespace()
        with patch.object(planner_factory.d2, 'prepare') as parent:
            planner_factory.prepare(owner, {}, {}, map_research=DEFAULT)
            parent.assert_called_once(); self.assertIsNone(owner.map_research_config)
        with patch.object(planner_factory.d2, 'build_controller', return_value='original') as parent:
            self.assertEqual(planner_factory.build_controller(owner, 'parent'), 'original')
            parent.assert_called_once_with(owner, 'parent')

    def test_capture_payload_schema_contains_no_hidden_state(self):
        # A schema allowlist catches accidental inclusion of target, obstacle,
        # layout, flow or full-scenario fields without needing a simulator.
        from dataclasses import fields
        self.assertEqual({f.name for f in fields(ScanObservation)},
                         {'origins', 'directions', 'distances', 'hit_mask', 'agent_ids'})
        self.assertEqual({f.name for f in fields(StepObservation)},
                         {'agent_start', 'agent_end', 'velocity_start', 'velocity_end',
                          'found_before', 'found_after', 'target_transition_completed',
                          'discovery_check_completed', 'episode_terminated', 'terminal_reason'})

    def test_no_new_events_after_terminal(self):
        j = journal(); j.append(1, 'team_search_detector', 'step', transition(collision=True))
        a = PublicObservationAdapter(); a.ingest(j.read())
        j.append(2, 'shared_neighbor26', 'scan', scan())
        with self.assertRaisesRegex(ValueError, 'after terminal'): a.ingest(j.read(a.cursor))

    def test_negative_check_requires_the_actual_scan(self):
        j = journal(); j.append(1, 'team_search_detector', 'step', transition())
        with self.assertRaisesRegex(ValueError, 'missing scan'): PublicObservationAdapter().ingest(j.read())

    def test_public_envelope_rejects_invalid_types(self):
        event, = journal().read()
        for bad in (replace(event, physical_step=True), replace(event, sequence=0),
                    replace(event, payload=None), replace(event, episode_id='')):
            with self.assertRaises(ValueError): PublicObservationAdapter().ingest((bad,))

    def test_reference_without_execute_never_reads_manifest_or_creates_output(self):
        import contextlib
        import io
        from pathlib import Path
        import uuid
        from project_paths import ROOT
        from chapter3_bser.experiments.coupled_search_v1.reference import main
        target = ROOT/'work'/('never_created_'+uuid.uuid4().hex)
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(['--config','missing','--interface-config',str(ROOT/'configs/coupled_search_v1.json'),
                           '--manifest','missing','--output',str(target)])
        self.assertFalse(result['executed']); self.assertFalse(target.exists())
        self.assertFalse(result['coupled_search_implemented'])
