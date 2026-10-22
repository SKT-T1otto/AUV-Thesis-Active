"""Cache invalidation, uncached algorithm parity and replay oracles; no episodes."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from core.env.public_events import ScanObservation, StepObservation, PublicEvent, payload_digest, vectors
from chapter3_bser.experiments.coupled_search_v1.geometry_cache import ObservationCache, box_key
from chapter3_bser.experiments.coupled_search_v1.geometry_hypotheses import GeometryHypothesis, GeometryHypothesisBank, RayConstraints, compatible
from chapter3_bser.experiments.coupled_search_v1.coupled_belief import CoupledBelief
from chapter3_bser.experiments.coupled_search_v1.core_config import CORE_DEFAULT
from chapter3_bser.experiments.coupled_search_v1.numerics import Budget, BudgetExceeded
from chapter3_bser.experiments.coupled_search_v1.types import PublicStepEvidence
from checks.map_cache_reference import ReferenceBank, matches as reference_matches
from checks.test_coupled_search import contract, specs, prior


def scan(distance=2., hit=True):
    return ScanObservation(((1.,1.,1.),), ((1.,0.,0.),), ((distance,),), ((hit,),), (0,))


def wall():
    return GeometryHypothesis((((4.,1.,1.),(2.,1.,1.)),))


def evaluate(cache, geometry, observation, **changes):
    args = dict(bounds=(20.,20.,10.), sensor_range=4., tolerance=.25)
    args.update(changes)
    return cache.compatible(geometry, observation, **args, evaluate=compatible)


class MapCache(unittest.TestCase):
    def test_accepted_entries_are_reused_and_snapshots_are_isolated(self):
        rays = RayConstraints([scan()], (20,20,10), 4, .25)
        geometry = wall(); box = geometry.boxes[0]
        with patch.object(rays, 'entry', wraps=rays.entry) as kernel:
            entry = rays.evaluate_entry(box); rays.remember_entry(box, entry)
            self.assertTrue(rays.matches(geometry)); self.assertTrue(rays.matches(geometry))
            self.assertEqual(kernel.call_count, 1)
        next_rays = RayConstraints([scan(3.)], (20,20,10), 4, .25)
        self.assertEqual(len(next_rays._entries.items), 0)
        self.assertFalse(next_rays.matches(geometry))

    def test_mutated_box_content_and_explicit_replacement_do_not_hit_old_entry(self):
        rays = RayConstraints([scan()], (20,20,10), 4, .25)
        box = (np.array([4.,1.,1.]), np.array([2.,1.,1.]))
        old_key = box_key(box)
        old = rays.cached_entry(box)
        box[0][0] += 1.
        new = rays.cached_entry(box)
        self.assertNotEqual(old_key, box_key(box))
        self.assertNotEqual(old.tobytes(), new.tobytes())
        rays.forget_entry(box)
        self.assertNotIn(box_key(box), rays._entries.items)

    def test_cached_geometry_checks_keep_the_final_sensor_free_guard(self):
        rays = RayConstraints([scan()], (20,20,10), 4, .25)
        enclosing = GeometryHypothesis((((1.,1.,1.),(.1,.1,.1)),))
        rays.remember_entry(enclosing.boxes[0], rays.entry(enclosing.boxes[0]))
        self.assertFalse(rays.matches(enclosing))

    def test_preprocessed_and_uncached_historical_rows_are_byte_identical(self):
        observations = [scan(), scan(4.,False), scan(), replace(scan(), origins=((2.,1.,1.),))]
        left = RayConstraints(observations, (20,20,10), 4, .25)
        cache = ObservationCache(8)
        right = RayConstraints(observations, (20,20,10), 4, .25, scan_cache=cache)
        for name in ('origins','directions','distances','hits','limits','parallel'):
            self.assertEqual(getattr(left,name).tobytes(), getattr(right,name).tobytes(), name)
        self.assertEqual(cache.counts['prepared_hits'],1)
        self.assertFalse(right.matches(wall()))  # contradictory miss is retained

    def test_prepared_snapshots_cannot_be_mutated(self):
        cache = ObservationCache(4)
        item = cache.prepare(scan(), (20,20,10), 4)
        for name in ('origins','directions','distances','hits','limits','parallel','rows'):
            array = getattr(item,name)
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        rays = RayConstraints([scan()], (20,20,10), 4, .25)
        with self.assertRaises(ValueError):
            rays.origins[0,0] = 100

    def test_completed_false_and_true_results_both_hit(self):
        cache = ObservationCache(4)
        for geometry in (wall(), GeometryHypothesis(())):
            first = evaluate(cache,geometry,scan())
            self.assertEqual(evaluate(cache,geometry,scan()), first)
        self.assertEqual(cache.counts['compatibility_misses'],2)
        self.assertEqual(cache.counts['compatibility_hits'],2)

    def test_scan_fields_do_not_alias_equal_steps_or_origins(self):
        cache = ObservationCache(8)
        self.assertTrue(evaluate(cache,wall(),scan()))
        self.assertFalse(evaluate(cache,wall(),scan(3.)))
        self.assertFalse(evaluate(cache,wall(),scan(4.,False)))
        self.assertFalse(evaluate(cache,wall(),replace(scan(),directions=((-1.,0.,0.),))))
        self.assertFalse(evaluate(cache,wall(),replace(scan(),origins=((0.,1.,1.),))))
        self.assertEqual(cache.counts['compatibility_misses'],5)

    def test_bounds_range_and_tolerance_changes_invalidate_compatibility(self):
        cache = ObservationCache(8)
        self.assertTrue(evaluate(cache,wall(),scan(2.2)))
        self.assertFalse(evaluate(cache,wall(),scan(2.2),tolerance=.1))
        self.assertFalse(evaluate(cache,wall(),scan(2.2),sensor_range=1.))
        self.assertFalse(evaluate(cache,wall(),scan(2.2),bounds=(2.,20.,10.)))
        self.assertEqual(cache.counts['compatibility_misses'],4)

    def test_geometry_keys_use_all_ordered_box_parameters(self):
        cache = ObservationCache(8)
        self.assertTrue(evaluate(cache,wall(),scan()))
        moved = GeometryHypothesis((((5.,1.,1.),(2.,1.,1.)),))
        self.assertFalse(evaluate(cache,moved,scan()))
        enlarged = GeometryHypothesis((((4.,1.,1.),(4.,1.,1.)),))
        self.assertFalse(evaluate(cache,enlarged,scan()))
        self.assertEqual(cache.counts['compatibility_misses'],3)

    def test_exceptions_are_never_published_as_compatibility(self):
        cache = ObservationCache(4)
        for error in (ValueError('invalid'), BudgetExceeded('wall_clock_budget')):
            with self.assertRaises(type(error)):
                cache.compatible(wall(),scan(),(20,20,10),4,.25,lambda *a,**k: (_ for _ in ()).throw(error))
            self.assertEqual(len(cache.compatibility.items),0)
        self.assertTrue(evaluate(cache,wall(),scan()))

    def test_byte_limits_and_eviction_change_only_recomputation(self):
        cache = ObservationCache(1,prepared_bytes=20000,compatibility_bytes=20000,compatibility_items=1)
        for observation in (scan(),scan(3.),scan()):
            self.assertEqual(evaluate(cache,wall(),observation),compatible(wall(),observation,(20,20,10),4,.25))
            self.assertLessEqual(cache.prepared.bytes,cache.prepared.max_bytes)
            self.assertLessEqual(cache.compatibility.bytes,cache.compatibility.max_bytes)
            self.assertLessEqual(len(cache.prepared.items),1)
        self.assertEqual(cache.counts['compatibility_misses'],3)
        no_cache = ObservationCache(1,prepared_bytes=0,compatibility_bytes=0)
        for _ in range(2):
            self.assertTrue(evaluate(no_cache,wall(),scan()))
        self.assertEqual(no_cache.counts['compatibility_hits'],0)
        self.assertEqual(no_cache.prepared.bytes,0)
        rays = RayConstraints([scan()],(20,20,10),4,.25,entry_cache_bytes=0)
        self.assertEqual(rays.matches(wall()),reference_matches(rays,wall()))
        self.assertEqual(rays._entries.bytes,0)

    def test_cache_hits_still_observe_each_original_budget_check(self):
        belief = prior()
        item = PublicStepEvidence('cache',0,(PublicEvent('cache',0,1,'shared_neighbor26','scan',scan(4.,False),'unused'),),None)
        belief._update(item,Budget(1))
        with patch.object(Budget,'check',side_effect=[None,BudgetExceeded('wall_clock_budget')]) as check:
            with self.assertRaises(BudgetExceeded):
                belief._update(item,Budget(1))
        self.assertEqual(check.call_count,2)
        self.assertGreater(belief.scan_cache.counts['compatibility_hits'],0)

    def test_new_episode_beliefs_do_not_share_caches(self):
        left = CoupledBelief(copy.deepcopy(CORE_DEFAULT),contract(),specs())
        right = CoupledBelief(copy.deepcopy(CORE_DEFAULT),contract(),specs())
        self.assertIs(left.scan_cache,left.geometry.scan_cache)
        self.assertIsNot(left.scan_cache,right.scan_cache)
        evaluate(left.scan_cache,wall(),scan())
        self.assertEqual(len(right.scan_cache.compatibility.items),0)

    def test_full_proposal_matches_uncached_oracle_and_rng_state(self):
        observations = [replace(scan(), origins=((1.,1.+i*.1,1.),)) for i in range(20)]
        banks = [ReferenceBank(copy.deepcopy(CORE_DEFAULT),contract(),np.random.default_rng(17)),
                 GeometryHypothesisBank(copy.deepcopy(CORE_DEFAULT),contract(),np.random.default_rng(17))]
        expected, actual = [bank.propose(observations,Budget(2)) for bank in banks]
        self.assertEqual([g.identity for g in expected],[g.identity for g in actual])
        self.assertEqual(banks[0].rng.bit_generator.state,banks[1].rng.bit_generator.state)
        self.assertLess(banks[1].last_entry_stats['kernel_calls'],banks[0].kernel_calls)

    def test_full_replay_preserves_posterior_and_rng_with_compatibility_cache(self):
        cfg = copy.deepcopy(CORE_DEFAULT)
        cached = CoupledBelief(cfg,contract(),specs()); reference = CoupledBelief(copy.deepcopy(cfg),contract(),specs())
        reference.scan_cache = ObservationCache(cfg['history_steps']+1,prepared_bytes=0,compatibility_bytes=0)
        reference.geometry = ReferenceBank(reference.config,reference.contract,reference.rng)
        for step in range(17):
            observation = replace(scan(3.),origins=((1.,2.+step*.1,1.),))
            event = PublicEvent('cache',step,step+1,'shared_neighbor26','scan',observation,payload_digest(observation))
            transition = None
            if step:
                positions = vectors([[18.,18.,9.]]*4); velocities = vectors(np.zeros((4,3)))
                payload = StepObservation(positions,positions,velocities,velocities,False,False,True,True,False,'running')
                transition = PublicEvent('cache',step,100+step,'team_search_detector','step',payload,payload_digest(payload))
            item = PublicStepEvidence('cache',step,(event,),transition)
            cached.consume(item); reference.consume(item)
            self.assertTrue(cached.valid,(step,cached.error))
            self.assertTrue(reference.valid,(step,reference.error))
            self.assertEqual(cached.identity,reference.identity)
            self.assertEqual(cached.rng.bit_generator.state,reference.rng.bit_generator.state)
        cached._rebuild(Budget(2)); reference._rebuild(Budget(2))
        self.assertEqual(cached.identity,reference.identity)
        self.assertEqual(cached.rng.bit_generator.state,reference.rng.bit_generator.state)

    def test_near_hit_merge_failure_is_preserved_without_algorithm_fix(self):
        observation = ScanObservation(((1.,1.,1.),),((1.,0.,0.),(-1.,0.,0.)),((.01,.01),),((True,True),),(0,))
        banks = [ReferenceBank(copy.deepcopy(CORE_DEFAULT),contract(),np.random.default_rng(17)),
                 GeometryHypothesisBank(copy.deepcopy(CORE_DEFAULT),contract(),np.random.default_rng(17))]
        for bank in banks:
            with self.assertRaisesRegex(ValueError,'constructed_geometry_does_not_match_public_rays'):
                bank.propose([observation],Budget(1))
        self.assertEqual(banks[0].rng.bit_generator.state,banks[1].rng.bit_generator.state)

    def test_failed_replay_keeps_previous_state_with_warm_caches(self):
        belief = prior(); before = belief.identity
        item = PublicStepEvidence('cache',0,(PublicEvent('cache',0,1,'shared_neighbor26','scan',scan(4.,False),'unused'),),None)
        belief.history.append(item);belief._update(item,Budget(1));before=belief.identity
        with patch.object(belief.geometry,'propose',return_value=belief.maps),patch.object(belief,'_propagate',side_effect=ValueError('replay_failed')):
            # A failure after an actual cached update must roll the state back.
            update=belief._update
            def fail(item,budget,**kwargs):
                update(item,budget,**kwargs)
                raise ValueError('replay_failed')
            with patch.object(belief,'_update',side_effect=fail):
                with self.assertRaisesRegex(ValueError,'replay_failed'):
                    belief._rebuild(Budget(2))
        self.assertEqual(belief.identity,before)
