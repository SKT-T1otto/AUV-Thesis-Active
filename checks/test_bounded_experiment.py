"""Experiment pairing, identity, incomplete-arm and explicit-execution checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from project_paths import ROOT
from chapter3_bser.learning import bounded_experiment as design
from chapter3_bser.learning import map_suite as suite


def matrix():return suite.read(ROOT/'configs/bounded_design_experiment_v1.json')


def prepared(directory):
    m=matrix();m['arms']={k:v for k,v in m['arms'].items() if k in ('reference_v2','full_w64')}
    m['stages']['pilot'].update(train_episodes=3,checkpoint_interval=1,evaluation_count=1)
    m['stages']['pilot']['convergence']['validation_count']=2
    spec=Path(directory)/'spec.json';spec.write_text(json.dumps(m),encoding='utf-8')
    root=Path(directory)/'collision_terminal/design';design.prepare(root,spec)
    return root


class BoundedExperiment(unittest.TestCase):
    def test_seven_arms_preserve_kernel_and_planning_scales(self):
        configs=design.specifications(matrix(),'pilot');self.assertEqual(len(configs),7)
        for spec in configs.values():
            c=spec['new_map_config']
            self.assertEqual((c['maps'],c['particles_per_map'],c['horizon'],c['execution_scenarios']),(3,32,16,3))
            self.assertEqual(spec['groups'],['NEW_ZERO','NEW_B2','NEW_B3'])

    def test_development_and_formal_task_generators_are_separate(self):
        stages=matrix()['stages'];p,f=stages['pilot'],stages['formal']
        self.assertNotEqual(p['evaluation_generator_seed'],f['evaluation_generator_seed'])
        self.assertNotEqual(p['convergence']['generator_seed'],f['convergence']['generator_seed'])
        self.assertNotEqual(p['training_generator_seed_base'],f['training_generator_seed_base'])

    def test_dry_run_does_not_read_spec_or_create_directories(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)/'missing'
            r=design.main(['prepare','--root',str(root),'--spec',str(root/'absent.json')])
            self.assertFalse(r['executed']);self.assertFalse(root.exists())

    def test_timing_bands_include_late_work(self):
        r=design.timing_bands([dict(seconds=x) for x in range(1,10)])
        self.assertEqual(r['early_p50'],2.);self.assertEqual(r['late_p50'],8.)
        self.assertIsNone(design.timing_bands([])['late_p95'])

    def test_prepared_arms_share_tasks_but_have_distinct_native_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            root=prepared(directory);value=design.load(root)
            plans=[suite.load_plan(root/arm) for arm in value['plans']]
            self.assertEqual(plans[0]['assets']['inputs/evaluation.json'],plans[1]['assets']['inputs/evaluation.json'])
            self.assertNotEqual(plans[0]['jobs'][1]['map_condition'],plans[1]['jobs'][1]['map_condition'])
            self.assertNotEqual(plans[0]['jobs'][1]['train'], '')
            self.assertFalse(any(root.rglob('checkpoint*.pt')))

    def test_missing_results_remain_incomplete_instead_of_being_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=prepared(directory);result=design.report(root,Path(directory)/'report')
            self.assertFalse(result['complete']);self.assertEqual(len(result['arms']),2)
            self.assertTrue(all(not row['complete'] and row['error'] for row in result['arms']))
            self.assertFalse(result['comparisons'])
