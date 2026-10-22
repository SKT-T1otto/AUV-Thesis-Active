"""Independent finite-state, posterior-boundary and failure oracles; no episodes."""
import copy
from dataclasses import replace
import unittest
import numpy as np
from core.env.public_events import PublicEvent,ScanObservation,StepObservation,payload_digest,vectors
from chapter3_bser.experiments.coupled_search_v1.bounded_config import BOUNDED_DEFAULT,validate_bounded_config
from chapter3_bser.experiments.coupled_search_v1.bounded_belief import BoundedCoupledBelief
from chapter3_bser.experiments.coupled_search_v1.persistent_map import PersistentVoxelMap,LocalGeometryBank
from chapter3_bser.experiments.coupled_search_v1.geometry_hypotheses import GeometryHypothesis
from chapter3_bser.experiments.coupled_search_v1.numerics import Budget
from chapter3_bser.experiments.coupled_search_v1.types import PublicStepEvidence
from chapter3_bser.experiments.coupled_search_v1.config import validate_research_config,search_planner_id
from chapter3_bser.learning.map_contract import validate_condition,identity
from checks.test_coupled_search import contract,specs


def config(window=16):
    c=copy.deepcopy(BOUNDED_DEFAULT);c['enabled']=True;c['history_steps']=window
    c['maps']=1;c['particles_per_map']=2;c['geometry']['unknown_boxes']=0
    c['bounded'].update(position_jitter=0.,velocity_jitter=0.,resample_ess_fraction=0.)
    return c


def scan(hit=False):
    return ScanObservation(((1.,1.,1.),),((1.,0.,0.),),((2. if hit else 4.,),),((hit,),),(0,))


def evidence(step,observation=None,*,negative=False,found=False,terminal=False):
    observation=scan() if observation is None else observation
    s=PublicEvent('bounded',step,step*2+1,'shared_neighbor26','scan',observation,payload_digest(observation))
    transition=None
    if step:
        p=vectors([[2.,1.,1.]]*4);v=vectors(np.zeros((4,3)))
        payload=StepObservation(p,p,v,v,False,found,negative,negative,terminal,'timeout' if terminal else 'running')
        transition=PublicEvent('bounded',step,step*2+2,'team_search_detector','step',payload,payload_digest(payload))
    return PublicStepEvidence('bounded',step,(s,),transition)


def analytic_belief(c=None):
    c=config() if c is None else c;ct=contract();ct['target_motion_mode']='static'
    belief=BoundedCoupledBelief(c,ct,specs());belief.consume(evidence(0))
    belief.initialize_hypotheses((GeometryHypothesis(()),),[[[2.,1.,1.],[10.,1.,1.]]],np.zeros((1,2,3)),[1.],[[.5,.5]])
    belief.boundary=belief._snapshot()
    return belief


class BoundedMap(unittest.TestCase):
    def test_v3_is_explicit_and_does_not_relabel_v2(self):
        self.assertEqual(validate_research_config(config())['schema'],'ch3.coupled_search.core.v3')
        self.assertEqual(search_planner_id(config()),'coupled_search_bounded_v3')
        from chapter3_bser.experiments.coupled_search_v1.core_config import CORE_DEFAULT
        self.assertEqual(search_planner_id(CORE_DEFAULT),'coupled_search_v2')

    def test_window_and_geometry_capacity_are_validated(self):
        for value in (1,15,65,256,1000):
            c=config();c['history_steps']=value
            with self.assertRaises(ValueError):validate_bounded_config(c)
        c=config();c['bounded']['max_cells']=100
        with self.assertRaisesRegex(ValueError,'cell budget'):validate_bounded_config(c)
        c=config();c['bounded']['max_boxes']=1000
        with self.assertRaises(ValueError):validate_bounded_config(c)

    def test_ablation_names_cannot_hide_different_switches(self):
        c=config();c['bounded']['boundary_mode']='uniform_ablation'
        with self.assertRaisesRegex(ValueError,'ablation label'):validate_bounded_config(c)
        c['bounded']['design_variant']='no_boundary'
        self.assertEqual(validate_bounded_config(c)['bounded']['boundary_mode'],'uniform_ablation')

    def test_raw_history_and_snapshots_stay_bounded_after_multiple_rollovers(self):
        b=analytic_belief()
        for step in range(1,71):
            b.consume(evidence(step));self.assertTrue(b.valid,b.error)
            self.assertLessEqual(len(b.history),16);self.assertLessEqual(len(b.snapshots),16)
            self.assertLessEqual(len(b.seen),42)
        self.assertEqual(b.boundary.step,54)
        self.assertEqual([x.physical_step for x in b.history],list(range(55,71)))

    def test_boundary_preserves_negative_mass_after_raw_evidence_leaves_window(self):
        b=analytic_belief();b.consume(evidence(1,negative=True))
        for step in range(2,19):b.consume(evidence(step))
        self.assertTrue(b.valid,b.error)
        self.assertGreater(b.boundary.step,1)
        np.testing.assert_array_equal(b.boundary.omega,[[0.,1.]])
        self.assertNotIn(1,[x.physical_step for x in b.history])

    def test_local_recovery_uses_positive_boundary_support_without_uniform_reset(self):
        b=analytic_belief();b.consume(evidence(1,negative=True))
        for step in range(2,20):b.consume(evidence(step))
        b.positions[:]=[2.,1.,1.]
        b.consume(evidence(20,negative=True))
        self.assertTrue(b.valid,b.error);self.assertEqual(b.counts['local_recoveries'],1)
        self.assertEqual(b.counts['uniform_boundary_ablation_resets'],0)
        np.testing.assert_array_equal(b.positions[0],[[10.,1.,1.],[10.,1.,1.]])
        self.assertEqual(b.last_replay_steps,16)

    def test_replay_bound_does_not_depend_on_episode_age(self):
        b=analytic_belief();b.consume(evidence(1,negative=True))
        for step in range(2,65):
            if step in (32,64):b.positions[:]=[2.,1.,1.]
            b.consume(evidence(step,negative=step in (32,64)))
            self.assertTrue(b.valid,b.error)
        self.assertEqual(b.counts['local_recoveries'],2)
        self.assertEqual(b.max_replay_steps,16)
        self.assertLessEqual(b.last_replay_work,2*16)

    def test_disabled_recovery_is_an_explicit_negative_control(self):
        c=config();c['bounded'].update(local_recovery=False,design_variant='no_local_recovery')
        b=analytic_belief(c);b.positions[:]=[2.,1.,1.];b.consume(evidence(1,negative=True))
        self.assertFalse(b.valid);self.assertIn('local_recovery_disabled_ablation',b.error)
        self.assertEqual(b.step,0)

    def test_no_positive_boundary_mass_fails_without_prior_fallback(self):
        b=analytic_belief()
        b.boundary=replace(b.boundary,alpha=np.array([0.]),omega=np.zeros((1,2)))
        b.positions[:]=[2.,1.,1.];b.consume(evidence(1,negative=True))
        self.assertFalse(b.valid);self.assertIn('boundary_proposal_free_space_exhausted',b.error)
        self.assertEqual(b.counts['uniform_boundary_ablation_resets'],0)

    def test_uniform_boundary_is_only_available_as_labeled_ablation(self):
        c=config();c['bounded'].update(boundary_mode='uniform_ablation',design_variant='no_boundary')
        b=analytic_belief(c);b.positions[:]=[2.,1.,1.];b.consume(evidence(1,negative=True))
        self.assertTrue(b.valid,b.error)
        self.assertGreater(b.counts['uniform_boundary_ablation_resets'],0)
        self.assertFalse(b.summary()['historical_information_carried_by_boundary'])

    def test_persistent_geometry_survives_old_scan_leaving_window(self):
        b=analytic_belief();b.consume(evidence(1,scan(True)))
        remote=replace(scan(),origins=((10.,10.,5.),))
        before=b.memory.classes().copy()
        self.assertTrue(np.any(before==1))
        for step in range(2,25):b.consume(evidence(step,remote))
        self.assertTrue(b.valid,b.error);self.assertTrue(np.all(b.memory.classes()[before==1]==1))

    def test_window_map_ablation_forgets_an_unrevisited_old_surface(self):
        c=config();c['bounded'].update(map_memory='window_ablation',design_variant='window_map')
        b=analytic_belief(c);b.consume(evidence(1,scan(True)))
        before=b.memory.classes().copy();remote=replace(scan(),origins=((10.,10.,5.),))
        for step in range(2,25):b.consume(evidence(step,remote))
        self.assertTrue(b.valid,b.error);self.assertTrue(np.all(b.memory.classes()[before==1]==0))
        self.assertLessEqual(len(b.map_frames),16)

    def test_coalesced_boxes_cover_exactly_the_declared_occupied_voxels(self):
        c=config();c['bounded']['grid_shape']=[4,4,4]
        memory=PersistentVoxelMap(c,contract());memory.logodds[1:3,1,1]=6;memory.logodds[0,3,2]=6
        boxes=memory.boxes();self.assertEqual(len(boxes),2)
        cells=np.array(list(np.ndindex(memory.shape)))
        centers=(cells+.5)*memory.spacing
        actual=~GeometryHypothesis(boxes).free(centers)
        expected=(memory.classes()==1).reshape(-1)
        np.testing.assert_array_equal(actual,expected)

    def test_box_budget_failure_never_drops_occupied_cells(self):
        c=config();c['bounded'].update(grid_shape=[4,4,4],max_boxes=4)
        memory=PersistentVoxelMap(c,contract());memory.logodds[::2,:,:]=6
        before=memory.logodds.copy()
        with self.assertRaisesRegex(ValueError,'box_capacity'):memory.boxes()
        np.testing.assert_array_equal(before,memory.logodds)

    def test_public_sensor_origin_cannot_be_enclosed_by_voxel_solids(self):
        c=config();memory=PersistentVoxelMap(c,contract())
        near=replace(scan(True),distances=((.01,),))
        memory.ingest([near],Budget(1))
        self.assertTrue(np.all(GeometryHypothesis(memory.boxes()).free(near.origins)))
        self.assertGreater(memory.hit_seen.sum(),0)

    def test_unknown_completion_cannot_fill_long_term_observed_cells(self):
        c=config();memory=PersistentVoxelMap(c,contract());memory.ingest([scan()],Budget(1))
        self.assertFalse(memory.unknown_box_allowed(((2.,1.,1.),(.5,.5,.5))))
        self.assertTrue(memory.unknown_box_allowed(((15.,15.,5.),(.5,.5,.5))))

    def test_invalid_sensor_range_is_rejected_before_large_allocation(self):
        memory=PersistentVoxelMap(config(),contract());before=memory.logodds.copy()
        with self.assertRaisesRegex(ValueError,'range_exceeds'):
            memory.ingest([replace(scan(),distances=((1e12,),))],Budget(1))
        np.testing.assert_array_equal(memory.logodds,before)
        self.assertEqual(memory.observed.sum(),0)

    def test_duplicate_packet_and_old_packet_after_window_eviction_are_safe(self):
        b=analytic_belief();item=evidence(1);b.consume(item);before=b.identity;b.consume(item)
        self.assertEqual(before,b.identity)
        for step in range(2,41):b.consume(evidence(step))
        with self.assertRaisesRegex(ValueError,'step_gap'):b.consume(item)

    def test_initial_scan_and_bounded_scan_batch_contract(self):
        b=BoundedCoupledBelief(config(),contract(),specs())
        with self.assertRaisesRegex(ValueError,'initial_public_scan'):b.consume(evidence(1))
        item=evidence(0)
        with self.assertRaisesRegex(ValueError,'batch_capacity'):
            b.consume(replace(item,scans=item.scans*2))

    def test_discovery_stops_future_search_consumption(self):
        b=analytic_belief();b.consume(evidence(1,found=True))
        self.assertTrue(b.stopped);identity_before=b.identity;b.consume(evidence(2))
        self.assertEqual(identity_before,b.identity)

    def test_map_migration_preserves_zero_mass_and_rejects_all_blocked_support(self):
        b=analytic_belief();b.omega[0]=[0.,1.]
        b._transport((GeometryHypothesis(()),))
        np.testing.assert_array_equal(b.omega,[[0.,1.]])
        with self.assertRaisesRegex(ValueError,'transport_zero_mass'):
            b._transport((GeometryHypothesis((((10.,1.,1.),(1.,1.,1.)),)),))

    def test_window_cannot_enable_the_v2_complete_history_guard(self):
        from chapter3_bser.learning.map_suite import validate_spec
        from pathlib import Path
        import json
        spec=json.loads((Path(__file__).resolve().parents[1]/'configs/map_comparison_v1.json').read_text())
        spec['new_map_config']=config();self.assertEqual(validate_spec(spec)['max_steps'],400)
        spec['new_map_config']['schema']='ch3.coupled_search.core.v2'
        with self.assertRaises(ValueError):validate_spec(spec)

    def test_native_map_identity_contains_the_v3_algorithm(self):
        from chapter3_bser.learning.checkpoint import digest
        condition=dict(schema='auv.active.map_condition.v1',map_id='NEW',research_config=config(),protocol_sha256='a'*64)
        value=identity(dict(map_experiment=condition))
        self.assertEqual(value['search_planner_id'],'coupled_search_bounded_v3')
        self.assertEqual(value['research_config_sha256'],digest(config()))
