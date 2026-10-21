"""Continuous sensor-evidence oracles; no simulator episodes or file restores."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from core.env.public_events import PublicEvent,ScanObservation,StepObservation,payload_digest,vectors
from core.env.target_motion import TargetState,advance_target_state
from chapter3_bser.experiments.coupled_search_v1.core_config import CORE_DEFAULT,validate_core_config
from chapter3_bser.experiments.coupled_search_v1.geometry_hypotheses import GeometryHypothesis,GeometryHypothesisBank,compatible
from chapter3_bser.experiments.coupled_search_v1.coupled_belief import CoupledBelief
from chapter3_bser.experiments.coupled_search_v1.target_particles import propagate
from chapter3_bser.experiments.coupled_search_v1.numerics import Budget
from chapter3_bser.experiments.coupled_search_v1.types import PublicStepEvidence
from chapter3_bser.experiments.coupled_search_v1.session import CoupledSearchSession
from checks.test_coupled_search import contract,specs,prior,evidence


def event(step,scan,sequence=1):
    return PublicEvent('persistence',step,sequence,'shared_neighbor26','scan',scan,payload_digest(scan))


class MapPersistence(unittest.TestCase):
    def bank(self):return GeometryHypothesisBank(copy.deepcopy(CORE_DEFAULT),contract(),np.random.default_rng(17))

    def test_health_gate_accepts_legitimate_early_discovery(self):
        from tools.check_map_training import continuous_health
        row=dict(actual_length=32,found_step=7,map_metrics=dict(search_batches=7,available_search_batches=7,
            unavailable_search_batches=0,core_scored_steps=6,last_core_scored_step=6))
        self.assertTrue(continuous_health(row))

    def test_health_gate_rejects_one_proposal_without_late_participation(self):
        from tools.check_map_training import continuous_health
        row=dict(actual_length=32,found_step=None,map_metrics=dict(search_batches=32,available_search_batches=32,
            unavailable_search_batches=0,core_scored_steps=1,last_core_scored_step=1))
        self.assertFalse(continuous_health(row))

    def test_health_gate_rejects_recovered_final_flag_after_unavailable_search(self):
        from tools.check_map_training import continuous_health
        row=dict(actual_length=32,found_step=None,map_metrics=dict(search_batches=32,available_search_batches=31,
            unavailable_search_batches=1,core_scored_steps=31,last_core_scored_step=31,final_filter_valid=True))
        self.assertFalse(continuous_health(row))

    def test_single_surface_more_than_eight_hits_is_supported(self):
        origins=tuple((1.,float(y),float(z)) for y in (2,4,6) for z in (2,4,6))
        scan=ScanObservation(origins,((1.,0.,0.),),tuple((3.,) for _ in origins),tuple((True,) for _ in origins),tuple(range(9)))
        bank=self.bank();maps=bank.propose([scan],Budget(5))
        self.assertEqual(len(maps),3)
        self.assertTrue(all(compatible(m,scan,contract()['bounds'],4.,.25) for m in maps))
        self.assertGreater(bank.support_boxes,0)

    def test_constructed_world_respects_every_distinct_historical_ray(self):
        scans=[ScanObservation(((1.,1.+i*.1,1.),),((1.,0.,0.),),((3.,),),((True,),),(0,)) for i in range(20)]
        maps=self.bank().propose(scans,Budget(5))
        for m in maps:
            for scan in scans:self.assertTrue(compatible(m,scan,contract()['bounds'],4.,.25))

    def test_free_ray_cannot_be_filled_by_unobserved_completions(self):
        scan=ScanObservation(((1.,1.,1.),),((1.,0.,0.),),((4.,),),((False,),),(0,))
        for m in self.bank().propose([scan],Budget(5)):
            self.assertTrue(compatible(m,scan,contract()['bounds'],4.,.25))

    def test_very_near_hit_shrinks_support_without_enclosing_sensor(self):
        scan=ScanObservation(((1.,1.,1.),),((1.,0.,0.),),((.01,),),((True,),),(0,))
        for m in self.bank().propose([scan],Budget(5)):
            self.assertTrue(np.all(m.free(scan.origins)))
            self.assertTrue(compatible(m,scan,contract()['bounds'],4.,.25))

    def test_contradictory_public_measurements_are_rejected(self):
        hit=ScanObservation(((1.,1.,1.),),((1.,0.,0.),),((1.,),),((True,),),(0,))
        miss=replace(hit,distances=((4.,),),hit_mask=((False,),))
        with self.assertRaisesRegex(ValueError,'constraints_inconsistent'):
            self.bank().propose([hit,miss],Budget(5))

    def test_old_geometry_prior_cannot_be_silently_relabelled(self):
        config=copy.deepcopy(CORE_DEFAULT);config['schema']='ch3.coupled_search.core.v1'
        with self.assertRaises(ValueError):validate_core_config(config)

    def test_free_flight_batch_is_exact_original_integrator(self):
        c=contract();g=GeometryHypothesis(())
        rng=np.random.default_rng(732);p=rng.uniform([1,1,1],[19,19,9],(32,3));v=rng.uniform(-.7,.7,(32,3))
        out,velocity=propagate(p,v,g,0,c,Budget(5))
        for i in range(32):
            expected=advance_target_state(TargetState(p[i],v[i],0,c['target_motion_mode']),c['dt'],c['bounds'],(),max_reflections=8)
            np.testing.assert_array_equal(out[i],expected.position);np.testing.assert_array_equal(velocity[i],expected.velocity)

    def test_batch_contacts_and_world_reflections_use_original_physics(self):
        c=contract();g=GeometryHypothesis((((5.,5.,5.),(2.,2.,2.)),))
        p=np.asarray([[3.9,5,5],[19.99,3,3],[.01,.01,1],[7,5,5]],float)
        v=np.asarray([[1,0,0],[1,0,0],[-1,-1,0],[-10,0,0]],float)
        for step in range(20):
            expected=[advance_target_state(TargetState(a,b,step,c['target_motion_mode']),c['dt'],c['bounds'],g.obstacles(),max_reflections=8) for a,b in zip(p,v)]
            p,v=propagate(p,v,g,step,c,Budget(5))
            np.testing.assert_array_equal(p,[e.position for e in expected]);np.testing.assert_array_equal(v,[e.velocity for e in expected])

    def test_failed_full_replay_restores_previous_joint_state(self):
        b=prior();p=b.positions.copy();v=b.velocities.copy();alpha=b.alpha.copy()
        b.history.append(PublicStepEvidence('persistence',0,(),None))
        with patch.object(b.geometry,'propose',return_value=b.maps),patch.object(b,'_update',side_effect=ValueError('negative_history_inconsistent')):
            with self.assertRaises(ValueError):b._rebuild(Budget(5))
        np.testing.assert_array_equal(b.positions,p);np.testing.assert_array_equal(b.velocities,v);np.testing.assert_array_equal(b.alpha,alpha)
        self.assertEqual(b.step,0)

    def test_unavailable_search_raises_instead_of_silent_fallback(self):
        session=CoupledSearchSession(copy.deepcopy(CORE_DEFAULT),specs(),contract())
        item=PublicStepEvidence('persistence',0,(),None)
        with patch('chapter3_bser.experiments.coupled_search_v1.audit.PublicEvidenceSession.consume',return_value=(item,)),patch.object(session.belief,'consume'):
            with self.assertRaisesRegex(RuntimeError,'coupled_search_unavailable'):session.consume(())
        self.assertEqual(session.counts['unavailable_search_batches'],1)

    def test_found_phase_stops_search_without_requiring_new_search_hypotheses(self):
        session=CoupledSearchSession(copy.deepcopy(CORE_DEFAULT),specs(),contract())
        item=evidence(found_after=True)
        with patch('chapter3_bser.experiments.coupled_search_v1.audit.PublicEvidenceSession.consume',return_value=(item,)),patch.object(session.belief,'consume'):
            session.consume(())
        self.assertEqual(session.counts['unavailable_search_batches'],0)

    def test_400_step_public_evidence_sequence_remains_usable(self):
        cfg=copy.deepcopy(CORE_DEFAULT);b=CoupledBelief(cfg,contract(),specs())
        positions=vectors([[1.,1.,1.],[18.,18.,9.],[18.,18.,9.],[18.,18.,9.]])
        velocities=vectors(np.zeros((4,3)))
        for step in range(401):
            # Distinct scans move slowly along one known flat surface. The
            # fixture supplies measurements, never a true geometry to b.
            y=2.+(step//20)*.1
            scan=ScanObservation(((1.,y,1.),),((1.,0.,0.),),((3.,),),((True,),),(0,))
            transition=None
            if step:
                payload=StepObservation(positions,positions,velocities,velocities,False,False,True,True,step==400,'timeout' if step==400 else 'running')
                transition=PublicEvent('persistence',step,step*2+1,'team_search_detector','step',payload,payload_digest(payload))
            b.consume(PublicStepEvidence('persistence',step,(event(step,scan,step*2+1),),transition))
            self.assertTrue(b.valid,(step,b.error));self.assertEqual(b.step,step)
            if step%50==0 and step<400:
                f=b.forecast(Budget(5));self.assertTrue(np.isfinite(f.positions).all());self.assertAlmostEqual(sum(f.weights),1.)
        self.assertTrue(b.history_complete);self.assertTrue(b.stopped)
        self.assertGreater(b.counts['full_history_replays'],1)
