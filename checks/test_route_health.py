"""Public-start feasibility and strict health/failure diagnostic oracles."""
import copy,json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from project_paths import ROOT
from chapter3_bser.experiments.coupled_search_v1.route_feasibility import feasible_groups
from chapter3_bser.experiments.coupled_search_v1.allocator import CoupledSearchAllocator
from chapter3_bser.learning import map_contract as mc


def state(statuses=('invalid_start','ready','ready')):
    agents=tuple(NS(agent_id=i,role='searcher' if i<3 else 'executor',position=(float(i*3),1.,1.)) for i in range(4))
    endpoints=[]
    for i,status in enumerate(statuses):
        if status=='stale_start_endpoint':continue
        endpoints.append(NS(role='searcher',point=agents[i].position,point_valid=status!='invalid_start',
            connectors=(object(),) if status=='ready' else ()))
    graph=NS(endpoint_connectors=endpoints,cell_centers=np.array([[15.,15.,5.]]),
        valid_mask=np.array([True]),component_labels=np.array([0]))
    return NS(agents=agents,searcher_ids=(0,1,2),planning_graph=graph,target_found=False)


def routes(ids=(1,2)):return tuple(NS(agent_id=i) for i in ids)


def runtime(directory):
    cfg=json.loads((ROOT/'configs/coupled_search_bounded_v3.json').read_text(encoding='utf-8'))
    condition=dict(schema=mc.SCHEMA,map_id='NEW',research_config=cfg,protocol_sha256='0'*64)
    counts=Counter(search_batches=32,available_search_batches=32,core_proposals=1,core_scored_steps=31)
    belief=NS(valid=True,first_failure=None,summary=lambda:dict(valid=True,error=None),timings=[])
    session=NS(valid=True,counts=counts,belief=belief,summary=lambda:dict(counts=dict(counts)))
    proposal=dict(kind='coupled_proposal',elapsed_seconds=.01,path_queries=1,compression={})
    return NS(config=dict(map_experiment=condition,output_dir=str(directory)),map_research_session=session,
        controller=NS(allocator=NS(audit=[proposal])),step=32,
        scenario={'scenario_id':'public_scene','obstacles':object()})


class RouteHealth(unittest.TestCase):
    def test_public_invalid_unassigned_start_is_held_while_other_agents_plan(self):
        s=state();active,held=feasible_groups(s,routes(),{0,1,2},set())
        self.assertEqual(active,{1,2});self.assertEqual(held,{'0':'invalid_start'});self.assertEqual(len(s.agents),4)

    def test_public_point_without_connector_is_also_explicitly_held(self):
        active,held=feasible_groups(state(('no_start_connector','ready','ready')),routes(),{0,1,2},set())
        self.assertEqual(active,{1,2});self.assertEqual(held,{'0':'no_start_connector'})

    def test_candidate_shortage_at_ready_start_remains_a_failure(self):
        with self.assertRaisesRegex(ValueError,'missing_public_safe_route'):
            feasible_groups(state(('ready',)*3),routes(),{0,1,2},set())

    def test_existing_assignment_cannot_be_discarded_by_the_new_constraint(self):
        with self.assertRaisesRegex(ValueError,'missing_public_safe_route'):
            feasible_groups(state(),routes(),{0,1,2},{0})

    def test_stale_endpoint_does_not_prove_public_unreachability(self):
        with self.assertRaisesRegex(ValueError,'missing_public_safe_route'):
            feasible_groups(state(('stale_start_endpoint','ready','ready')),routes(),{0,1,2},set())

    def test_no_active_search_group_remains_a_visible_failure(self):
        with self.assertRaisesRegex(ValueError,'missing_public_safe_route'):
            feasible_groups(state(('invalid_start',)*3),(),{0,1,2},set())

    def test_all_available_groups_are_unchanged(self):
        self.assertEqual(feasible_groups(state(('ready',)*3),routes((0,1,2)),{0,1,2},{0,1,2}),({0,1,2},{}))

    def test_partial_recovery_keeps_atomic_rejection_for_unassigned_affected_agent(self):
        a=CoupledSearchAllocator.__new__(CoupledSearchAllocator);a.research={'schema':'ch3.coupled_search.core.v3'}
        old=object();a._propose=lambda *args:NS(search_assignments=(NS(agent_id=1),NS(agent_id=2)))
        result=a.allocate_partial(state(),old,affected_searcher_ids=(0,),trigger_reason='SAFE_SEARCH_RECOVERY')
        self.assertIs(result[0],old);self.assertEqual(result[1:],(False,'ATOMIC_REJECT_MISSING_SEARCH_ROUTE'))

    def test_successful_partial_recovery_preserves_the_original_return_contract(self):
        a=CoupledSearchAllocator.__new__(CoupledSearchAllocator);a.research={'schema':'ch3.coupled_search.core.v3'}
        proposed=NS(search_assignments=(NS(agent_id=0),NS(agent_id=1)));a._propose=lambda *args:proposed
        result=a.allocate_partial(state(),object(),affected_searcher_ids=(0,),trigger_reason='SAFE_SEARCH_RECOVERY')
        self.assertIs(result[0],proposed);self.assertEqual(result[1:],(True,'COUPLED_ATOMIC_PROPOSAL'))

    def test_v2_keeps_its_original_missing_route_rejection(self):
        a=CoupledSearchAllocator.__new__(CoupledSearchAllocator);a.research={'schema':'ch3.coupled_search.core.v2'}
        s=state();a._context=lambda *_:dict(state=s,targets=None,geometry=None,budget=None);a._references=lambda *_:{}
        with patch('chapter3_bser.experiments.coupled_search_v1.allocator.generate_routes',return_value=routes()),patch(
                'chapter3_bser.experiments.coupled_search_v1.route_feasibility.feasible_groups',side_effect=AssertionError('v3 hook in v2')):
            with self.assertRaisesRegex(ValueError,'missing_public_safe_route'):a._propose(s,None,{0,1,2},True,'INITIALIZE')

    def test_no_core_proposal_still_fails_and_saves_the_exact_clause(self):
        with tempfile.TemporaryDirectory() as directory:
            r=runtime(directory);r.map_research_session.counts['core_proposals']=0
            with self.assertRaisesRegex(RuntimeError,'no_core_proposal'):mc.metrics(r)
            p=next(Path(directory).glob('map_failure_*.json'));value=json.loads(p.read_text())
            self.assertEqual(value['phase'],'episode_health');self.assertEqual(value['details']['violations'],['no_core_proposal'])
            self.assertEqual(value['scenario_id'],'public_scene');self.assertNotIn('obstacles',value)

    def test_missing_late_scores_are_not_accepted_as_healthy(self):
        with tempfile.TemporaryDirectory() as directory:
            r=runtime(directory);r.map_research_session.counts['core_scored_steps']=1
            with self.assertRaisesRegex(RuntimeError,'missing_core_scored_steps'):mc.metrics(r)
            self.assertEqual(len(list(Path(directory).glob('map_failure_*.json'))),1)

    def test_healthy_map_metrics_create_no_failure_file(self):
        with tempfile.TemporaryDirectory() as directory:
            r=runtime(directory);value=mc.metrics(r)
            self.assertEqual(value['map_metrics']['core_proposals'],1);self.assertFalse(list(Path(directory).iterdir()))

    def test_advance_failure_records_public_audit_and_preserves_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            r=runtime(directory)
            def fail(*args,**kwargs):raise RuntimeError('original_exception')
            r.advance=fail
            with self.assertRaisesRegex(RuntimeError,'original_exception'):mc.advance(r,None,explore=False)
            value=json.loads(next(Path(directory).glob('map_failure_*.json')).read_text())
            self.assertEqual(value['phase'],'advance');self.assertEqual(len(value['allocator_audit_tail']),1)

    def test_diagnostic_failure_cannot_mask_the_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            r=runtime(directory);r.map_research_session.summary=lambda:(_ for _ in ()).throw(ValueError('logging_error'))
            r.advance=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('original_exception'))
            with self.assertRaisesRegex(RuntimeError,'original_exception'):mc.advance(r,None,explore=False)
            self.assertFalse(list(Path(directory).iterdir()))
