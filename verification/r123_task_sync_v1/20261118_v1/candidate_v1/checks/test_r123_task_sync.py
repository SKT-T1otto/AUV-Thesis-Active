"""Task receipt contracts and real allocator dataflow, without native envs."""
import copy
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np
import torch
from checks.test_r123_repair import fixture as parent_fixture
from chapter3_bser.r123_task_sync.receipts import snapshot
from chapter3_bser.r123_task_sync.allocator import SyncAllocator
from chapter3_bser.r123_task_sync.runtime import RuleRuntime
from chapter3_bser.r123_tracking.runtime import RuleRuntime as Tracking
from chapter3_bser.path_option.path_context import Route
from chapter3_bser.online.types import SearchAssignment,ExecutorAssignment,OnlineAllocation


def fixture():
    r,s,t=parent_fixture('no_recovery');r.__class__=RuleRuntime
    r.task_sync_enabled=True;r.sync_receipts=[];r.sync_scoring=[];r.sync_seconds=0.;r.sync_route_queries=0
    r.tracking_arm='both';r.tracking_switches=dict(path=True,feedforward=True);r.route_events=[]
    def observe(step=0):
        s.step=t.step=step;r._cached=None
        with patch('chapter3_bser.r123_tracking.runtime.extract_planning_state',return_value=s):return r.observe()
    observe()
    return r,s,t,observe


def allocator(r):
    from chapter3_bser.config import load_bser_phase1a1_config
    a=SyncAllocator(r,load_bser_phase1a1_config(),'D2',(9.,3.,4.))
    items=tuple(SearchAssignment(i,f'proposal{i}',(15.,3.,4.),((1.,3.,4.),(15.,3.,4.)),10.,(),10.) for i in range(3))
    allocation=OnlineAllocation(items,ExecutorAssignment(3,(9.,3.,4.),((9.,3.,4.),),0.,'test',True,(),0.),1.,1.,1.,'TEST')
    r.controller.inner.current_allocation=allocation
    return a,allocation


class SyncContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_proposal_does_not_force_adoption_or_reset_identity(self):
        r,s,t,observe=fixture();identity=r.lifecycle.task_id(0)
        item=r.controller.inner.current_allocation.search_assignments[0]
        item.waypoint=(15.,3.,4.);item.candidate_id='next'
        row=observe(1)[0]
        self.assertEqual(r.lifecycle.task_id(0),identity)
        np.testing.assert_equal(row['task'],[9.,3.,4.]);np.testing.assert_equal(row['original'],[15.,3.,4.])
        receipt=snapshot(r,s)[0];self.assertEqual(receipt.root,(9.,3.,4.));self.assertEqual(receipt.status,'ACTIVE')

    def test_completed_root_stays_released_until_new_token(self):
        r,s,t,observe=fixture();identity=r.lifecycle.task_id(0)
        r.lifecycle.release(0,0,'TASK_COMPLETE')
        for step in (1,10,20):self.assertEqual(observe(step)[0]['reason'],'WAIT_TASK')
        self.assertEqual(r.lifecycle.task_id(0),identity)
        item=r.controller.inner.current_allocation.search_assignments[0];item.candidate_id='new';item.waypoint=(15.,3.,4.)
        observe(21);self.assertNotEqual(r.lifecycle.task_id(0),identity)
        self.assertEqual(snapshot(r,s)[0].root,(15.,3.,4.));self.assertNotIn(0,r._release_requests)

    def test_unassigned_and_frozen_do_not_create_tasks(self):
        r,s,t,observe=fixture();r.controller.inner.current_allocation.search_assignments=()
        observe(1);self.assertTrue(all(r.lifecycle.tasks[i].released for i in range(3)))
        count=len([v for v in r.lifecycle_events if v['event']=='TASK_ACCEPT'])
        s.target_found=True;observe(2);observe(20)
        self.assertEqual(count,len([v for v in r.lifecycle_events if v['event']=='TASK_ACCEPT']))
        self.assertTrue(all(v.status=='FROZEN' for v in snapshot(r,s).values()))

    def test_released_unreachable_and_connector_never_credit_old_root(self):
        r,s,t,observe=fixture();r.lifecycle.release(0,0,'NO_ACTIVE_RECOVERY')
        r._cached=None
        p=np.array(s.agents[0].position);c=r.manager.current[0]
        r.manager.commit(0,c.original,c.task,p+[0,1,0],0,0,Route([p,p+[0,1,0]],1,0,True),reason='CONNECTOR')
        r.manager.current[0].mode='CONNECTOR'
        rec=snapshot(r,s)[0];self.assertEqual(rec.status,'CONNECTOR');self.assertNotEqual(rec.endpoint,rec.root)
        r.manager.current[0].route.reachable=False
        self.assertEqual(snapshot(r,s)[0].points,(tuple(p),))
        r.manager.current[0].mode='TASK';r.lifecycle.tasks[0].released=False
        self.assertEqual(snapshot(r,s)[0].status,'UNREACHABLE')
        self.assertEqual(snapshot(r,s)[0].points,(tuple(p),))

    def test_receipt_is_copy_and_keeps_full_route_without_trimming(self):
        r,s,t,observe=fixture();rec=snapshot(r,s)[0];points=rec.points
        r.manager.current[0].route.points[0]=99
        self.assertEqual(rec.points,points);self.assertNotEqual(rec.points,tuple(map(tuple,r.manager.current[0].route.points)))

    def test_rescore_uses_actual_path_and_preserves_proposal(self):
        r,s,t,observe=fixture();a,allocation=allocator(r);seen=[]
        def context(state,search,*args):seen.extend(search);return NS()
        with patch('chapter3_bser.d2.allocator.build_objective_context',side_effect=context),patch('chapter3_bser.d2.allocator.evaluate_objective',return_value=.5),patch('chapter3_bser.d2.allocator.response_diagnostics',return_value=NS(conditional_reachable_response_time=2.)):
            scored=a.rescore(s,allocation)
        self.assertEqual(scored.search_assignments,allocation.search_assignments)
        self.assertEqual(seen[0].waypoint,(9.,3.,4.))
        np.testing.assert_equal(seen[0].path_points,r.manager.current[0].route.points)

    def test_proposal_freezes_active_but_released_agent_gets_new_choice(self):
        r,s,t,observe=fixture();a,allocation=allocator(r);r.lifecycle.release(2,0,'TASK_COMPLETE')
        choices=tuple(a._frozen_search_candidate(v) for v in allocation.search_assignments);captured={}
        def parent(state,candidates,frozen,*args):
            captured['choices']=candidates;captured['fixed']=frozen
            return replace(allocation,search_assignments=tuple(a._search_assignment(v) for v in (*frozen,*candidates)))
        with patch('chapter3_bser.d2.allocator.EffectAllocator._proposal',side_effect=parent):
            result=a._proposal(s,choices,(),(),None,'TEST')
        self.assertEqual([v.agent_id for v in captured['choices']],[2])
        self.assertEqual([v.waypoint for v in captured['fixed']],[(9.,3.,4.)]*2)
        self.assertEqual(result.search_assignments[:2],allocation.search_assignments[:2])

    def test_released_fixed_other_has_hold_receipt_in_finalize(self):
        r,s,t,observe=fixture();a,allocation=allocator(r);r.lifecycle.release(1,0,'TASK_COMPLETE')
        a.sync_fixed_context=(s.step,frozenset({1}));seen=[]
        with patch('chapter3_bser.d2.allocator.build_objective_context',side_effect=lambda state,search,*args:(seen.extend(search) or NS())),patch('chapter3_bser.d2.allocator.evaluate_objective',return_value=.5),patch('chapter3_bser.d2.allocator.response_diagnostics',return_value=NS(conditional_reachable_response_time=2.)):
            a.rescore(s,allocation,finalize=True)
        self.assertEqual(len(seen[1].path_points),1)
        self.assertEqual(seen[1].waypoint,tuple(s.agents[1].position))

    def test_disabled_rescore_is_exact_parent_and_no_receipts_or_rng_use(self):
        r,s,t,observe=fixture();a,allocation=allocator(r);r.task_sync_enabled=False
        with patch('chapter3_bser.d2.allocator.EffectAllocator.rescore',return_value=allocation) as parent:
            self.assertIs(a.rescore(s,allocation),allocation)
        parent.assert_called_once_with(s,allocation,finalize=False);self.assertFalse(r.sync_scoring)
        state=np.random.get_state();snapshot(r,s);after=np.random.get_state()
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(state,after)))


if __name__=='__main__':unittest.main()
