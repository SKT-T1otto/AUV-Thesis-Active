"""Old-map executor geometry, recovery and actual RL-mask oracles; no episodes."""
import copy
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
from checks.test_executor_standby import fixture,SPEC,CONTRACT
from chapter3_bser.safety.executor_recovery import ExecutorRecoveryBridge,check_motion
from chapter3_bser.safety.executor_recovery_options import OPTIONS,validate_options
from chapter3_bser.safety.executor_public import ExecutorGeometry,QueryView,query_route
from chapter3_bser.safety.public_geometry import PublicGeometry
from chapter3_bser.learning import map_contract
from chapter3_bser.learning.runtime import RepairedMissionRuntime


def setup(*,occupied=False,known=True,position=(1.,1.5,1.5)):
    owner,state,allocation,context=fixture(occupied=occupied)
    owner.config['map_experiment']['executor_safety']=dict(OPTIONS)
    owner.planning_views.clearance=.4
    state.agents[3].position=position
    free=~state.occupancy.occupied_mask if known else np.zeros(len(state.grid.cell_centers),bool)
    owner.env.unwrapped=NS(map_module=NS(known_free_mask=torch.as_tensor(free.copy()),map_revision=0))
    return owner,state,allocation,context


class ExecutorRecovery(unittest.TestCase):
    def test_v1_and_disabled_original_identity_preserved(self):
        from chapter3_bser.safety.executor_standby import OPTIONS as V1
        self.assertEqual(validate_options(V1),V1)
        c=dict(schema=map_contract.SCHEMA,map_id='OLD',research_config=None,protocol_sha256='a'*64)
        self.assertEqual(map_contract.validate_condition(c),c)
        for value in (dict(OPTIONS,require_known_free=False),dict(OPTIONS,motion_steps=400),dict(OPTIONS,tracking_distance=float('nan'))):
            with self.assertRaises(ValueError):validate_options(value)

    def test_v2_rejects_new_map_and_changes_native_condition(self):
        from chapter3_bser.learning.map_suite import validate_spec
        import json
        from project_paths import ROOT
        c=dict(schema=map_contract.SCHEMA,map_id='NEW',research_config={},protocol_sha256='a'*64,executor_safety=OPTIONS)
        with self.assertRaisesRegex(ValueError,'OLD map'):map_contract.validate_condition(c)
        spec=json.loads((ROOT/'configs/executor_recovery_pilot_v2.json').read_text(encoding='utf-8'))
        self.assertEqual(validate_spec(spec)['groups'],['OLD_ZERO'])
        spec['groups']=['NEW_ZERO']
        with self.assertRaisesRegex(ValueError,'OLD-only'):validate_spec(spec)

    def test_known_free_segment_matches_independent_unknown_box_oracle(self):
        owner,state,_,_=setup();rng=np.random.default_rng(991)
        mask=rng.random(len(state.grid.cell_centers))>.35
        owner.env.unwrapped.map_module.known_free_mask=torch.as_tensor(mask)
        geometry=ExecutorGeometry(owner)
        oracle=PublicGeometry(NS(grid=state.grid,occupancy=NS(occupied_mask=~mask)),0.)
        for _ in range(60):
            p=rng.uniform([0,0,0],[10,4,4]);q=p+rng.uniform(-.8,.8,3);q=np.clip(q,[0,0,0],[10,4,4])
            self.assertEqual(geometry.known_segment(p,q),oracle.segment_free(p,q))
        self.assertEqual(geometry.known_segment([1,1,1],[1,1,1]),oracle.point_free([1,1,1]))

    def test_unknown_space_does_not_count_as_safe_empty_space(self):
        owner,state,allocation,context=setup(known=False)
        bridge=ExecutorRecoveryBridge(owner)
        route=NS(reachable=False,failure_reason='no_public_route')
        with patch('chapter3_bser.safety.executor_recovery.query_route',return_value=(route,0)):
            actual=bridge.compile_guidance(allocation,state,context)
        self.assertTrue(actual.executor_assignment.hold_state)
        self.assertEqual(bridge.mode,'UNVERIFIED')
        self.assertFalse(bridge.summary()['safety_guarantee'])

    def test_query_view_supplies_goal_without_mutating_public_targets(self):
        from core.env.task_state import AgentStateView
        original=AgentStateView(('search','search','search','execute'),((1.,1.,1.),)*4,
            ((0.,0.,0.),)*4,((2.,2.,2.),)*4,(False,)*4)
        env=NS(get_agent_state=lambda:original)
        view=QueryView(env,(8.,1.,1.));actual=view.get_agent_state()
        self.assertEqual(actual.navigation_targets[3],(8.,1.,1.))
        self.assertEqual(original.navigation_targets[3],(2.,2.,2.))
        self.assertEqual(actual.positions,original.positions)

    def test_route_query_uses_fresh_public_endpoints_without_provider_refresh(self):
        owner,_,_,_=setup();state=NS(step=7,agents=(NS(agent_id=3,position=(1,2,3)),),executor_id=3)
        owner.provider=NS(snapshot=lambda **k:(_ for _ in ()).throw(AssertionError('provider mutated')))
        with patch('chapter3_bser.safety.executor_public.extract_planning_state',return_value=state) as snapshot,patch(
             'chapter3_bser.safety.executor_public.filter_planning_state',side_effect=lambda s,c:s),patch(
             'chapter3_bser.safety.executor_public.TravelCostService') as service:
            route,step=query_route(owner,(8,2,3))
        self.assertEqual(step,7);snapshot.assert_called_once()
        self.assertEqual(snapshot.call_args.args[0].goal,(8,2,3))
        service.return_value.query.assert_called_once_with((1,2,3),(8,2,3),state.agents[0])

    def test_fixed_recovery_anchor_survives_agent_motion(self):
        owner,state,allocation,context=setup(occupied=True,position=(4.6,1.5,1.5))
        allocation=replace(allocation,executor_assignment=replace(allocation.executor_assignment,
            path=((4.6,1.5,1.5),(8.,1.5,1.5))))
        bridge=ExecutorRecoveryBridge(owner);route=NS(reachable=False,failure_reason='no_route')
        with patch('chapter3_bser.safety.executor_recovery.query_route',return_value=(route,0)) as query:
            first=bridge.compile_guidance(allocation,state,context)
            self.assertIn(bridge.mode,('RECOVER','HOLD'));anchor=bridge.anchor
            self.assertLess(anchor[0],state.agents[3].position[0])
            from chapter3_bser.experiments.coupled_search_v1.execution_forecast import nominal_step
            agent=state.agents[3]
            p,v=nominal_step(agent.position,agent.velocity,anchor,SPEC,bridge.contract)
            agent.position,agent.velocity=tuple(p),tuple(v);state.step=1
            second=bridge.compile_guidance(allocation,state,replace(context,step=1))
        self.assertEqual(bridge.anchor,anchor)
        self.assertEqual(second.executor_assignment.hold_position,anchor)
        self.assertNotEqual(anchor,agent.position);self.assertEqual(query.call_count,1)
        self.assertEqual(first.agent_assignments[:3],second.agent_assignments[:3])

    def test_public_velocity_increment_is_estimated_without_flow_truth(self):
        owner,state,allocation,context=setup();bridge=ExecutorRecoveryBridge(owner)
        first=bridge.compile_guidance(allocation,state,context)
        from chapter3_bser.experiments.coupled_search_v1.execution_forecast import nominal_step
        agent=state.agents[3];p,v=nominal_step(agent.position,agent.velocity,first.agent_assignments[3].tracking_waypoint,SPEC,bridge.contract)
        error=np.array([.02,-.01,0.]);agent.position=tuple(p+error*.2);agent.velocity=tuple(v+error);state.step=1
        bridge.compile_guidance(allocation,state,replace(context,step=1))
        np.testing.assert_allclose(bridge.error,error,atol=1e-12)
        self.assertEqual(bridge.counts['public_error_observations'],1)

    def test_bridge_keeps_allocation_searchers_and_executor_views_consistent(self):
        owner,state,allocation,context=setup();saved=copy.deepcopy(allocation)
        from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
        old=RMADDPGGuidanceBridge().compile_guidance(allocation,state,context)
        new=ExecutorRecoveryBridge(owner).compile_guidance(allocation,state,context)
        self.assertEqual(old.agent_assignments[:3],new.agent_assignments[:3]);self.assertEqual(saved,allocation)
        a,b=new.agent_assignments[3],new.executor_assignment
        self.assertEqual((a.tracking_waypoint,a.hold_position,a.hold_state,a.reachable),(b.tracking_waypoint,b.hold_position,b.hold_state,b.reachable))

    def test_safe_original_tracking_is_not_globally_slowed_or_rethresholded(self):
        owner,state,allocation,context=setup()
        from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
        original=RMADDPGGuidanceBridge().compile_guidance(allocation,state,context)
        bridge=ExecutorRecoveryBridge(owner)
        actual=bridge.compile_guidance(allocation,state,context)
        self.assertEqual(actual.agent_assignments[3],original.agent_assignments[3])
        self.assertGreater(np.linalg.norm(np.asarray(actual.agent_assignments[3].tracking_waypoint)-state.agents[3].position),OPTIONS['tracking_distance'])

    def test_found_target_returns_original_context_and_marks_exit(self):
        owner,state,allocation,context=setup();bridge=ExecutorRecoveryBridge(owner)
        bridge.compile_guidance(allocation,state,context)
        state.step=1;state.target_found=True;mission=replace(context,step=1,target_found=True,executor_knows_target=True)
        from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
        result=bridge.compile_guidance(allocation,state,mission)
        self.assertEqual(result,RMADDPGGuidanceBridge().compile_guidance(allocation,state,mission))
        self.assertEqual(bridge.trace[-1]['mode'],'HANDOFF_EXIT')

    def test_route_failures_are_throttled_and_storage_bounded(self):
        owner,state,allocation,context=setup(known=False);bridge=ExecutorRecoveryBridge(owner)
        route=NS(reachable=False,failure_reason='no_start_connector')
        with patch('chapter3_bser.safety.executor_recovery.query_route',side_effect=lambda *a:(route,state.step)) as query:
            for t in range(7):
                state.step=t;bridge.compile_guidance(allocation,state,replace(context,step=t))
        self.assertEqual(query.call_count,2)
        self.assertEqual(bridge.trace.maxlen,64);self.assertEqual(bridge.errors.maxlen,8);self.assertEqual(bridge.safe_positions.maxlen,8)

    def test_phase_gate_masks_only_executor_and_reenables_original_execution(self):
        r=RepairedMissionRuntime.__new__(RepairedMissionRuntime)
        r.config={'map_experiment':{'executor_safety':dict(OPTIONS)}}
        r.repair_config={'repairs':{'hold_residual':True,'inactive_actor':True}}
        r.repair_counts={};r.observations=np.zeros((4,28))
        r.guidance=NS(agent_assignments=tuple(NS(agent_id=i,reachable=True,hold_state=False) for i in range(4)))
        task=NS(target_found=False);r.env=NS(get_task_state=lambda:task)
        np.testing.assert_array_equal(r._gates(),[1,1,1,0])
        task.target_found=True;np.testing.assert_array_equal(r._gates(),[1,1,1,1])
        task.target_found=False;r.config['map_experiment']['executor_safety']['enabled']=False
        np.testing.assert_array_equal(r._gates(),[1,1,1,1])

    def test_actual_b2_b3_actor_and_replay_respect_search_gate_then_resume(self):
        from checks.test_learning import fresh,transition,params,same
        from chapter3_bser.learning.replay import ExecutionReplay
        for label in ('b2','b3'):
            model=fresh(label);before=params(model.agents[3].policy)
            replay=ExecutionReplay(4)
            for i in range(4):
                row=transition(i);row['actions'][3]=0.;row['action_enabled'][3]=False;row['next_action_enabled'][3]=False
                replay.push_transition(row)
            sample=replay.sample(4)
            for agent in range(4):model.update(sample,agent)
            model.update_all_targets()
            self.assertEqual(model.actor_updates,[1,1,1,0]);self.assertTrue(same(before,params(model.agents[3].policy)))
            original=ExecutionReplay(4)
            for i in range(4):original.push_transition(transition(i))
            sample=original.sample(4)
            for agent in range(4):model.update(sample,agent)
            model.update_all_targets()
            self.assertEqual(model.actor_updates,[2,2,2,1]);self.assertFalse(same(before,params(model.agents[3].policy)))


if __name__=='__main__':unittest.main()
