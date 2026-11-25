import hashlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from tools import run_launch_development_pilot as runner
from tools import launch_development_pilot_audit as audit
from tools.launch_development_pilot_audit import pilot_decision


class PilotTests(unittest.TestCase):
    def test_runtime_local_schedule_collects_repeat_zero_in_new_directory(self):
        plan = runner.read(runner.LOCAL_OUT / 'plan.json')
        jobs = list(runner.cells(plan, runner.LOCAL_OUT))
        self.assertEqual(len(jobs) * len(plan['arms']), 48)
        self.assertFalse(any(archived for _, _, archived, _ in jobs))
        self.assertEqual(jobs[0][3], runner.LOCAL_OUT / '0003_r0')
        self.assertTrue(all(folder.parent == runner.LOCAL_OUT for _, _, _, folder in jobs))

    def test_small_historical_differences_are_reported_exactly(self):
        reference = dict(initial={}, prefix=dict(menus=[dict(boat=0, features=[0.], points=[[1., 2., 3.]])]))
        snapshot = copy.deepcopy(reference)
        snapshot['prefix']['menus'][0]['points'][0][2] += 2**-22
        result = runner.historical_comparison(snapshot, reference)
        self.assertTrue(result['initial_equal'])
        self.assertFalse(result['menus_equal'])
        self.assertEqual(result['coordinates'][0]['max_coordinate_difference_m'], 2**-22)
        snapshot['prefix']['menus'][0]['points'].append([0., 0., 0.])
        result = runner.historical_comparison(snapshot, reference)
        self.assertFalse(result['coordinates'][0]['same_shape'])
        self.assertIsNone(result['coordinates'][0]['max_coordinate_difference_m'])

    def test_preflight_freezes_local_candidates_and_rejects_changed_resume(self):
        plan = dict(families=[dict(scene='0003')])
        sample = dict(scene='0003', menus=[1., 2.])
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'file_hash', return_value='hash'), \
                patch.object(runner, 'checked_source', return_value='source'), \
                patch.object(runner, 'runtime_identity', return_value=dict(system='Linux')), \
                patch.object(runner, 'preview', return_value=sample), patch.object(runner, 'collect_command') as collect:
            out = Path(tmp)
            runner.freeze_runtime(plan, out)
            runner.freeze_runtime(plan, out)
            sample['menus'][0] += 2**-22
            with self.assertRaisesRegex(ValueError, 'local frozen departures changed'):
                runner.freeze_runtime(plan, out)
            collect.assert_not_called()

    def test_schedule_reuses_only_the_historical_block(self):
        plan = runner.read(runner.OUT / 'plan.json')
        jobs = list(runner.cells(plan))
        self.assertEqual([j[0][-4:] for j in jobs[::4]], ['0003', '0020', '0014'])
        self.assertEqual([r for _, r, _, _ in jobs], [0, 1, 2, 3] * 3)
        self.assertEqual([j[3] for j in jobs if j[2]], [runner.ROOT / plan['baseline']])
        self.assertEqual(len({j[3] for j in jobs}), 12)
        self.assertEqual(sum(not j[2] for j in jobs) * len(plan['arms']), 44)

    def test_portable_import_resumes_and_rejects_conflicts_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'inputs.zip'
            payload = {'first': b'one', 'other/second': b'two'}
            with ZipFile(archive, 'w') as packed:
                for name, data in payload.items():
                    packed.writestr(name, data)
            expected = {n: hashlib.sha256(v).hexdigest() for n, v in payload.items()}
            dest = root / 'workspace'
            (dest / 'other').mkdir(parents=True)
            (dest / 'other/second').write_bytes(b'existing')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                runner.install(archive, expected, dest)
            self.assertFalse((dest / 'first').exists())
            self.assertEqual((dest / 'other/second').read_bytes(), b'existing')
            clean = root / 'new'
            runner.install(archive, expected, clean)
            runner.install(archive, expected, clean)
            self.assertEqual((clean / 'first').read_bytes(), b'one')
            with ZipFile(archive, 'a') as packed:
                packed.writestr('../extra', 'extra')
            with self.assertRaisesRegex(ValueError, 'members'):
                runner.install(archive, expected, root / 'unused')
            self.assertFalse((root / 'unused').exists())

    def test_a_tampered_input_cannot_be_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'bad.zip'
            with ZipFile(archive, 'w') as packed:
                packed.writestr('input', b'altered')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                runner.install(archive, {'input': hashlib.sha256(b'original').hexdigest()}, root)
            self.assertFalse((root / 'input').exists())

    def test_zero_and_discordant_families_can_trigger_both_decisions(self):
        def family(name, found, repeats):
            return dict(scene=name, counts=dict(found=found, n=16), discordant_repeats=repeats)
        rows = [family('a', 0, []), family('b', 8, [2]), family('c', 7, [])]
        result = pilot_decision(rows)
        self.assertEqual(result['low_discovery_families'], ['a', 'c'])
        self.assertEqual(result['success_discordant_blocks'], {'b': [2]})
        self.assertTrue(result['expand_paired_validation'] and result['inspect_search_coverage'])
        result = pilot_decision([family(name, 0, []) for name in 'abc'])
        self.assertFalse(result['expand_paired_validation'])
        self.assertTrue(result['inspect_search_coverage'])

    def test_matching_is_within_repeat_and_keeps_native_rejections(self):
        arms = ['00', '01', '10', '11']
        family = dict(scene='scene_0020', family='registered')
        plan = dict(arms=arms, repeats=[0, 1, 2, 3], cap=400, source_lock='lock',
                    actor_sha256='actor', continuation='policy', terminal='native')
        # Each of the first two arms succeeds once, in different repeats.
        winners = {0: '01', 1: '00'}
        def audited(folder):
            repeat = int(folder.name[-1])
            return dict(arms={arm: dict(labels=dict(censored=False, S=winners.get(repeat) == arm,
                                ever_found=True, ever_received=True), admissions=[dict(accepted=False)])
                              for arm in arms})
        def receipt(path):
            if path.name == 'runtime.json':
                snapshot = receipt(runner.OUT / '0020_r0/00.json')
                return dict(plan_sha256='receipt', source_lock='lock',
                            departures=[dict(**family, snapshot=snapshot, prefix_sha256=audit.family_prefix(snapshot))])
            repeat = int(path.parent.name[-1])
            if path.name == 'request.json':
                return dict(**family, repeat=repeat, cap=400, fixture=False, order=[0, 1, 2, 3],
                            source_lock='lock', actor_reference=dict(actor_parameters_sha256='actor'),
                            continuation='policy', terminal='native')
            if path.name == 'audit.json':
                return audited(path.parent)
            return dict(initial={}, actor_sha256='actor', protocol={},
                        prefix=dict(menus=[], draws=[], streams_before=[['o', 'e', repeat]] * 3,
                                    streams_after=[['o1', 'e1', repeat]] * 3),
                        episode=dict(frames=[dict(commands=[path.stem], routes=[], roots=[])], decisions=[]))
        geom = dict(centers_ever_covered=1, centers_until_first_continuation=1, continuation_requests=0)
        with patch.object(audit, 'audit_folder', side_effect=audited), patch.object(audit, 'read', side_effect=receipt), \
                patch.object(audit, 'file_hash', return_value='receipt'), patch.object(audit, 'coverage', return_value=geom):
            result = audit.one_family(plan, family, dict(scenario_id=family['scene'], planner_grid_size=[1]*3,
                                      obstacles=[]), runner.OUT)
            local = audit.one_family(dict(plan, reuse_archived_baseline=False), family,
                                     dict(scenario_id=family['scene'], planner_grid_size=[1]*3, obstacles=[]), runner.LOCAL_OUT)
            self.assertEqual(local['counts'], result['counts'])
            with patch.object(audit, 'family_prefix', side_effect=['frozen', 'frozen', 'changed']):
                with self.assertRaisesRegex(ValueError, 'branch differs'):
                    audit.one_family(dict(plan, reuse_archived_baseline=False), family,
                                     dict(scenario_id=family['scene'], planner_grid_size=[1]*3, obstacles=[]), runner.LOCAL_OUT)
        self.assertEqual(result['counts'], dict(n=16, found=16, received=16, success=2))
        self.assertEqual(result['discordant_repeats'], [0, 1])
        self.assertEqual(result['paired_success_contrasts']['00-01'], dict(first_only=1, second_only=1, equal=2))
        self.assertEqual(result['by_arm']['11']['success'], 0)

    def test_unavailable_offers_do_not_become_failed_assignments(self):
        plan = dict(families=[dict(scene=s) for s in ('one', 'two')])
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'OUT', Path(tmp)), \
                patch.object(runner, 'preview', side_effect=lambda _, f: dict(**f, menu_sizes=[1, 2, 2],
                            four_arms_available=False)), patch.object(runner, 'collect_command') as collect, \
                patch.object(runner, 'publish_summary') as summary:
            with self.assertRaisesRegex(SystemExit, 'one, two'):
                runner.run(plan, 1)
            collect.assert_not_called()
            summary.assert_not_called()
            self.assertEqual(len(list(Path(tmp).glob('*_unavailable.json'))), 2)


if __name__ == '__main__':
    unittest.main()
