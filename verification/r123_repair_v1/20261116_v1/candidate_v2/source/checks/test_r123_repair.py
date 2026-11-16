"""Combination regressions from the six supplied Linux rule episodes; no env."""
import copy
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np
import torch
from checks.test_r123 import runtime_fixture,observe
from chapter3_bser.r123.options import CONTROL
from chapter3_bser.path_option.path_context import Route
from chapter3_bser.r123_repair.lifecycle import Lifecycle,ProgressManager
from chapter3_bser.r123_repair.runtime import RuleRuntime
from chapter3_bser.r123_repair.planning import connector,independent_replans


def fixture(variant='repaired'):
    r,state,task=runtime_fixture();r.__class__=RuleRuntime
    r.variant=variant;r.lifecycle=Lifecycle(CONTROL);r.lifecycle_events=r.lifecycle.events
    r.manager=ProgressManager(r.o,r.lifecycle,True)
    r._task_tokens={};r._connectors={};r._last_root_query={};r.trace=[];r._explore=False
    allocation=NS(search_assignments=tuple(NS(agent_id=i,candidate_id=f'goal{i}',waypoint=(9.,3.,4.),failure_reason=None) for i in range(3)),
                  executor_assignment=NS(target_region=(9.,3.,4.),source='standby'))
    r.controller=NS(inner=NS(current_allocation=allocation))
    r.planning_views=NS(search=lambda state:state)
    for i,a in enumerate(state.agents):a.role='searcher' if i<3 else 'executor'
    state.planning_graph=NS(endpoint_connectors=tuple(NS(role=a.role,point=a.position,point_valid=True,connectors=(1,)) for a in state.agents),
        valid_mask=np.ones(800,bool),component_labels=np.zeros(800,int),cell_centers=state.grid.cell_centers)
    from chapter3_bser.integration.control_context import AgentAssignmentContextV1 as A,ExecutorAssignmentContextV1 as E,BSERControlContextV1 as C
    assignments=tuple(A(i,'searcher' if i<3 else 'executor','search','test',(9.,3.,4.),(),(9.,3.,4.),(9.,3.,4.),False,True,False) for i in range(4))
    r.guidance=C('bser.control_context.v1','v','h',0,'SEARCH',assignments,E(3,'test',(9.,3.,4.),(),(9.,3.,4.),(9.,3.,4.),False,True,False),'TEST')
    r.env.install_guidance=lambda c:None
    return r,state,task


def tick(r,state,task,step):
    state.step=step;task.step=step;r._cached=None
    return observe(r,state)


def act(r):
    with patch('chapter3_bser.r123.runtime.RepairedMissionRuntime._actions_for_step',return_value=None):
        return r._actions_for_step(torch.zeros((4,3)))


class RepairContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_stall_recovery_next_step_keeps_same_attempt_alive(self):
        r,s,t=fixture();observe(r,s)
        rows=tick(r,s,t,30);self.assertEqual(rows[0]['reason'],'STALLED');act(r)
        root=r.lifecycle.tasks[0];identity=root.identity
        self.assertEqual(root.recoveries,1);self.assertEqual(r.manager.current[0].mode,'RECOVER')
        for step in (31,32,40,59):
            row=tick(r,s,t,step)[0]
            self.assertEqual(row['reason'],'CONTINUE');self.assertFalse(row['forced_terminate']);act(r)
        self.assertEqual(root.identity,identity);self.assertEqual(root.recoveries,1)
        self.assertEqual(root.progress_step,0)

    def test_two_genuinely_stalled_attempts_then_release(self):
        r,s,t=fixture();observe(r,s);tick(r,s,t,30);act(r)
        row=tick(r,s,t,60)[0]
        row['rule_goal']=next(j for j in np.flatnonzero(row['goal_mask']) if np.linalg.norm(r._candidates[0][j][0]-row['task'])>1e-4)
        act(r);self.assertEqual(r.lifecycle.tasks[0].recoveries,2)
        row=tick(r,s,t,61)[0];self.assertEqual(row['reason'],'CONTINUE')
        row=tick(r,s,t,90)[0];self.assertEqual(row['reason'],'WAIT_TASK')
        self.assertEqual(r.lifecycle.tasks[0].release_reason,'RECOVERY_EXHAUSTED')

    def test_duplicate_commit_does_not_reset_attempt_or_total_budget(self):
        r,s,t=fixture();observe(r,s);tick(r,s,t,30);act(r)
        c=r.manager.current[0];a=r.lifecycle.attempts[0]
        r.lifecycle.enter_recovery(0,50,c.endpoint.copy())
        r.manager.commit(0,c.original,c.task,c.endpoint,50,0,c.route,reason='REPEAT')
        self.assertIs(r.lifecycle.attempts[0],a)
        self.assertEqual(a.started,30);self.assertEqual(r.lifecycle.tasks[0].recovery_start,30)
        self.assertEqual(tick(r,s,t,60)[0]['reason'],'STALLED')

    def test_attempt_progress_does_not_extend_total_recovery_limit(self):
        r,s,t=fixture();observe(r,s);tick(r,s,t,30);act(r)
        r.lifecycle.attempts[0].progress_step=179
        row=tick(r,s,t,180)[0]
        self.assertEqual(row['reason'],'WAIT_TASK');self.assertTrue(r.lifecycle.tasks[0].released)

    def test_actual_recovery_arrival_returns_to_root(self):
        r,s,t=fixture();observe(r,s);c=r.manager.current[0];point=np.array(s.agents[0].position)
        root=c.task.copy();r.lifecycle.enter_recovery(0,0,point)
        r.manager.commit(0,c.original,c.task,point,0,0,Route([point],1,0,True),reason='RECOVER')
        tick(r,s,t,1)
        self.assertEqual(r.manager.current[0].mode,'TASK');np.testing.assert_equal(r.manager.current[0].endpoint,root)
        self.assertEqual(r.lifecycle.tasks[0].recoveries,1);self.assertNotIn(0,r.lifecycle.attempts)
        self.assertTrue(any(e['event']=='RECOVERY_RETURN' for e in r.lifecycle.events))

    def test_unassigned_hold_is_not_a_completed_task(self):
        r,s,t=fixture();r.controller.inner.current_allocation.search_assignments=()
        observe(r,s);tick(r,s,t,200)
        self.assertFalse(any(e['event'] in ('TASK_ACCEPT','TASK_RELEASE') and e.get('agent')<3 for e in r.lifecycle.events))
        self.assertTrue(all(r.lifecycle.tasks[i].identity==0 for i in range(3)))

    def test_same_completed_assignment_cannot_mint_a_new_task(self):
        r,s,t=fixture();observe(r,s);root=r.lifecycle.tasks[0]
        r.lifecycle.release(0,0,'TASK_COMPLETE');row=tick(r,s,t,100)[0]
        self.assertEqual(row['reason'],'WAIT_TASK');self.assertEqual(r.lifecycle.tasks[0].identity,root.identity)

    def test_new_assignment_adopted_after_release(self):
        r,s,t=fixture();observe(r,s);old=r.lifecycle.task_id(0);r.lifecycle.release(0,0,'TASK_COMPLETE')
        item=r.controller.inner.current_allocation.search_assignments[0]
        item.candidate_id='new';item.waypoint=(15.,3.,4.)
        tick(r,s,t,1)
        self.assertNotEqual(r.lifecycle.task_id(0),old);self.assertNotIn(0,r._release_requests)
        np.testing.assert_equal(r.manager.current[0].task,[15.,3.,4.])

    def test_hold_guidance_does_not_replace_semantic_assignment(self):
        r,s,t=fixture()
        from dataclasses import replace
        r.guidance=replace(r.guidance,agent_assignments=tuple(replace(a,final_waypoint=s.agents[a.agent_id].position,hold_state=True,reachable=False) for a in r.guidance.agent_assignments))
        rows=observe(r,s)
        np.testing.assert_equal(rows[0]['original'],[9.,3.,4.]);self.assertEqual(rows[0]['assignment_kind'],'search')

    def test_exhausted_task_can_track_root_without_release_resume_loop(self):
        r,s,t=fixture();observe(r,s);root=r.lifecycle.tasks[0];identity=root.identity
        root.recoveries=2;r.lifecycle.release(0,1,'RECOVERY_EXHAUSTED')
        for step in (100,101,110):
            row=tick(r,s,t,step)[0]
            self.assertEqual(row['reason'],'TRACK_ROOT');self.assertFalse(row['rule_terminate'])
        self.assertEqual(root.identity,identity);self.assertEqual(root.recoveries,2);self.assertFalse(root.released)

    def test_frozen_searchers_do_not_query_or_recover_or_move(self):
        r,s,t=fixture();observe(r,s);s.target_found=True
        with patch.object(r,'_query',wraps=r._query) as query:
            rows=tick(r,s,t,200);actual=act(r)
        self.assertTrue(all(call.args[0]==3 for call in query.call_args_list))
        for row in rows[:3]:self.assertEqual(row['reason'],'FROZEN');self.assertFalse(row['rule_terminate'])
        self.assertFalse(any(e['event']=='RECOVERY_ENTER' and e['agent']<3 for e in r.lifecycle.events))
        np.testing.assert_equal(actual.numpy()[:3],np.zeros((3,3)))

    def test_failed_recovery_can_return_to_root_without_immediate_new_detour(self):
        r,s,t=fixture();observe(r,s);tick(r,s,t,30);act(r)
        row=tick(r,s,t,60)[0]
        row['rule_goal']=next(j for j in np.flatnonzero(row['goal_mask']) if np.linalg.norm(r._candidates[0][j][0]-row['task'])<1e-4)
        act(r)
        self.assertTrue(r.lifecycle.tasks[0].root_only)
        self.assertEqual(tick(r,s,t,61)[0]['reason'],'TRACK_ROOT')
        self.assertEqual(r.lifecycle.tasks[0].recoveries,1)

    def test_no_recovery_has_no_alternatives_or_attempts(self):
        r,s,t=fixture('no_recovery');observe(r,s)
        for step in (30,31,60,100):tick(r,s,t,step);act(r)
        self.assertFalse(r._alternatives);self.assertFalse(r.lifecycle.attempts)
        self.assertFalse(any(e['event']=='RECOVERY_ENTER' for e in r.lifecycle.events))

    def test_no_recovery_cannot_train(self):
        with self.assertRaisesRegex(ValueError,'cannot train'):
            RuleRuntime({},None,learner=object(),variant='no_recovery')

    def test_one_failed_pool_does_not_block_other_agent(self):
        r,s,t=fixture();calls=[];installed=[]
        def allocate(state,current,*,affected_searcher_ids,**kwargs):
            calls.append(affected_searcher_ids);i=affected_searcher_ids[0]
            return NS(version=i),i==1,'OK' if i==1 else 'MISSING_ROUTE'
        inner=NS(allocator=NS(allocate_partial=allocate),current_allocation=NS(version=-1))
        def install(p,i):installed.append(i);inner.current_allocation=p
        rows=independent_replans(inner,s,[0,1],install)
        self.assertEqual(calls,[(0,),(1,)]);self.assertEqual(installed,[1]);self.assertEqual([x['accepted'] for x in rows],[False,True])

    def test_invalid_start_is_isolated_before_allocator_call(self):
        r,s,t=fixture();s.planning_graph.endpoint_connectors[0].point_valid=False
        inner=NS(allocator=NS(allocate_partial=lambda *a,**kw:(NS(),True,'OK')),current_allocation=NS())
        installed=[];rows=independent_replans(inner,s,[0,1],lambda p,i:installed.append(i))
        self.assertEqual(rows[0]['reason'],'invalid_start');self.assertEqual(installed,[1])

    def test_connector_is_public_free_valid_and_does_not_edit_state(self):
        r,s,t=fixture();s.planning_graph.endpoint_connectors[0].point_valid=False
        before=s.planning_graph.valid_mask.tobytes(),s.occupancy.free_mask.tobytes(),copy.deepcopy(s.agents[0].position)
        route=connector(s,0,NS(segment=lambda p,q:True))
        self.assertIsNotNone(route);self.assertLessEqual(np.linalg.norm(route.points[-1]-s.agents[0].position),3.)
        self.assertEqual(before,(s.planning_graph.valid_mask.tobytes(),s.occupancy.free_mask.tobytes(),s.agents[0].position))
        self.assertIsNone(connector(s,0,NS(segment=lambda p,q:False)))

    def test_wait_can_execute_public_connector_without_recovery_budget(self):
        r,s,t=fixture('no_recovery');observe(r,s)
        s.planning_graph.endpoint_connectors[0].point_valid=False
        r.lifecycle.release(0,0,'NO_ACTIVE_RECOVERY')
        row=tick(r,s,t,1)[0]
        self.assertEqual(row['reason'],'CONNECTOR');self.assertFalse(row['rule_terminate'])
        self.assertEqual(r.lifecycle.tasks[0].recoveries,0);self.assertEqual(r.manager.current[0].mode,'CONNECTOR')

    def test_wait_uses_same_filtered_start_view_as_d2_replanning(self):
        r,s,t=fixture('no_recovery');observe(r,s)
        filtered=copy.deepcopy(s);filtered.planning_graph.endpoint_connectors[0].point_valid=False
        r.planning_views=NS(search=lambda state:filtered)
        r.lifecycle.release(0,0,'NO_ACTIVE_RECOVERY')
        row=tick(r,s,t,1)[0]
        self.assertEqual(row['reason'],'CONNECTOR')
        self.assertTrue(s.planning_graph.endpoint_connectors[0].point_valid)

    def test_wait_preserves_verified_buffer_exit(self):
        r,s,t=fixture();observe(r,s);r.lifecycle.release(0,0,'TASK_COMPLETE')
        with patch('chapter3_bser.r123_repair.runtime.refined_buffer_exit',return_value=(np.array([.1,0,0]),{})):
            row=tick(r,s,t,1)[0]
        self.assertEqual(row['reason'],'WAIT_TASK');self.assertTrue((row['forced']==9).all())
        self.assertTrue(row['action_mask'][:,9].all());self.assertFalse(row['action_mask'][:,8].any())

    def test_cli_has_no_training_command_and_previews_do_not_mutate(self):
        from tools.r123_rule_experiment import main
        with patch('tools.r123_rule_experiment.prepare',side_effect=AssertionError('mutation')):
            self.assertFalse(main(['prepare','--output','unused'])['executed'])
        with self.assertRaises(SystemExit):main(['run'])


if __name__=='__main__':unittest.main()
