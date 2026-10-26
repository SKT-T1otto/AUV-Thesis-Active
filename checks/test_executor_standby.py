"""Public route, inertial braking, identity and baseline controls; no episodes."""
import copy
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from chapter3_bser.safety.executor_standby import OPTIONS, ExecutorStandbyBridge, motion_check, validate_options
from chapter3_bser.safety.public_geometry import PublicGeometry
from chapter3_bser.safety.safe_guidance import SafetyBridge
from chapter3_bser.online.types import ExecutorAssignment, OnlineAllocation
from chapter3_bser.online.mission_context import OnlineMissionContext
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.learning import map_contract
from chapter3_bser import planner_factory
from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.map_suite import validate_spec
from chapter3_bser.learning.standby_experiment import episode_record

SPEC = dict(a_xy_max=.8,a_z_max=.55,v_xy_max=1.5,v_z_max=.75,drag_xy=.16,drag_z=.24,buoyancy_bias=.02)
CONTRACT = dict(dt=.2,eps=1e-6,use_residual_prior=True,prior_kv_xy=1.1,prior_kv_z=1.,
    prior_slow_radius_xy=2.4,prior_slow_radius_z=1.2,prior_strength_search=.2,prior_strength_executor=1.,bounds=(10.,4.,4.))


def fixture(*,occupied=True,velocity=(0.,0.,0.)):
    centers=np.array(list(np.ndindex((10,4,4))),float)+.5
    mask=np.zeros(len(centers),bool)
    if occupied:mask[np.all(centers==[5.5,1.5,1.5],axis=1)]=True
    agents=tuple(NS(agent_id=i,role='execute' if i==3 else 'search',position=(1.,1.5,1.5),
                    velocity=velocity,current_navigation_target=(8.,1.5,1.5)) for i in range(4))
    state=NS(step=0,target_found=False,mission_complete=False,executor_id=3,searcher_ids=(0,1,2),agents=agents,
        grid=NS(shape=(10,4,4),origin=(0.,0.,0.),spacing=(1.,1.,1.),cell_centers=centers),occupancy=NS(occupied_mask=mask))
    executor=ExecutorAssignment(3,(8.,1.5,1.5),((1.,1.5,1.5),(8.,1.5,1.5)),1.,'standby',True)
    allocation=OnlineAllocation((),executor,0.,0.,0.,'test')
    context=OnlineMissionContext(0,False,-1,False,None,False,None,(False,)*4,(False,)*3)
    owner=NS(config={'map_experiment':{'executor_safety':dict(OPTIONS)}},safe_counts=Counter(),safe_options={'path_safety':False},
        state=state,env=NS(get_public_vehicle_specs=lambda:[SPEC]*4,get_public_control_contract=lambda:CONTRACT),
        planning_views=NS(clearance=0.,live=lambda:NS(step=state.step,agents=state.agents,grid=state.grid,occupancy=state.occupancy),search=lambda s:s))
    return owner,state,allocation,context


class ExecutorStandby(unittest.TestCase):
    def test_disabled_and_absent_use_original_bridge_and_old_identity(self):
        owner,state,allocation,context=fixture(occupied=False)
        owner.config['map_experiment'].pop('executor_safety')
        self.assertIs(type(planner_factory.build_bridge(owner)),SafetyBridge)
        c=dict(schema=map_contract.SCHEMA,map_id='OLD',research_config=None,protocol_sha256='a'*64)
        expected=dict(schema=c['schema'],map_id='OLD',search_planner_id='original_d2',condition_sha256=digest(c),
                      research_config_sha256=digest(None),protocol_sha256='a'*64)
        # Lock the historical identity exactly, including the original ID string.
        from chapter3_bser.experiments.coupled_search_v1.config import search_planner_id
        expected['search_planner_id']=search_planner_id(None)
        self.assertEqual(map_contract.identity({'map_experiment':c}),expected)
        owner.config['map_experiment']['executor_safety']=dict(OPTIONS,enabled=False)
        self.assertIs(type(planner_factory.build_bridge(owner)),SafetyBridge)
        owner.config['map_experiment']['executor_safety']['enabled']=True
        self.assertIsInstance(planner_factory.build_bridge(owner),ExecutorStandbyBridge)

    def test_guard_changes_identity_and_checkpoint_binding(self):
        c=dict(schema=map_contract.SCHEMA,map_id='OLD',research_config=None,protocol_sha256='a'*64)
        off={'map_experiment':dict(c,executor_safety=dict(OPTIONS,enabled=False))}
        on={'map_experiment':dict(c,executor_safety=dict(OPTIONS))}
        self.assertNotEqual(map_contract.identity(off),map_contract.identity(on))
        with self.assertRaises(ValueError):map_contract.validate_checkpoint_map(on,{'map_condition':map_contract.identity(off)})
        for value in (dict(OPTIONS,motion_steps=400),dict(OPTIONS,enabled=1),dict(OPTIONS,braking_steps=0)):
            with self.assertRaises(ValueError):validate_options(value)

    def test_clear_route_preserves_searcher_context_and_does_not_query(self):
        owner,state,allocation,context=fixture(occupied=False)
        expected=RMADDPGGuidanceBridge().compile_guidance(allocation,state,context)
        bridge=ExecutorStandbyBridge(owner)
        with patch('chapter3_bser.safety.executor_standby.TravelCostService') as query:
            actual=bridge.compile_guidance(allocation,state,context)
        self.assertEqual(actual,expected);query.assert_not_called()
        self.assertEqual(bridge.counts['checks'],1)

    def test_blocked_route_brakes_and_keeps_both_executor_views_consistent(self):
        owner,state,allocation,context=fixture()
        frozen=copy.deepcopy(allocation)
        bridge=ExecutorStandbyBridge(owner)
        route=NS(reachable=False,failure_reason='no_public_route')
        with patch('chapter3_bser.safety.executor_standby.TravelCostService') as service:
            service.return_value.query.return_value=route
            actual=bridge.compile_guidance(allocation,state,context)
        service.return_value.query.assert_called_once()
        executor=actual.agent_assignments[3]
        self.assertTrue(executor.hold_state);self.assertFalse(executor.reachable)
        self.assertEqual(executor.hold_position,state.agents[3].position)
        self.assertEqual(actual.executor_assignment.hold_state,executor.hold_state)
        self.assertEqual(actual.executor_assignment.planned_path,executor.planned_path)
        self.assertEqual(actual.agent_assignments[:3],RMADDPGGuidanceBridge().compile_guidance(allocation,state,context).agent_assignments[:3])
        self.assertEqual(allocation,frozen)
        self.assertEqual(bridge.summary()['last_search_guidance']['route_failure'],'no_public_route')

    def test_repair_retained_then_invalidated_by_assignment_and_latest_geometry(self):
        owner,state,allocation,context=fixture()
        bridge=ExecutorStandbyBridge(owner)
        path=((1.,1.5,1.5),(1.,3.5,1.5),(8.,3.5,1.5),(8.,1.5,1.5))
        route=NS(reachable=True,failure_reason=None,path_points=path,path_cell_indices=(),physical_travel_time=9.,planning_cost=9.)
        with patch('chapter3_bser.safety.executor_standby.TravelCostService') as service:
            service.return_value.query.return_value=route
            repaired=bridge.compile_guidance(allocation,state,context)
            self.assertFalse(repaired.executor_assignment.hold_state)
            self.assertEqual(repaired.executor_assignment.planned_path,path)
            bridge.compile_guidance(allocation,state,context)
            self.assertEqual(service.return_value.query.call_count,1)
            # Same semantic assignment, newly occupied route must not reuse validation.
            state.occupancy.occupied_mask[:]=True
            held=bridge.compile_guidance(allocation,state,context)
            self.assertTrue(held.executor_assignment.hold_state)
            self.assertIsNone(bridge._executor_override)
            self.assertEqual(service.return_value.query.call_count,2)
        bridge._executor_override=(('old',),allocation.executor_assignment)
        state.target_found=True
        bridge.compile_guidance(allocation,state,replace(context,target_found=True))
        self.assertIsNone(bridge._executor_override)

    def test_found_handoff_and_complete_preserve_original_context(self):
        owner,state,allocation,context=fixture()
        bridge=ExecutorStandbyBridge(owner)
        for found,known,complete in ((True,False,False),(True,True,False),(True,True,True)):
            state.target_found=found
            ctx=replace(context,target_found=found,executor_knows_target=known,mission_complete=complete)
            with patch('chapter3_bser.safety.executor_standby.TravelCostService') as service:
                actual=bridge.compile_guidance(allocation,state,ctx)
            expected=RMADDPGGuidanceBridge().compile_guidance(allocation,state,ctx)
            self.assertEqual(actual,expected);service.assert_not_called()
        self.assertEqual(bridge.counts['checks'],0)

    def test_inertial_hold_is_not_instantaneous_stop_or_safety_guarantee(self):
        owner,state,allocation,context=fixture()
        geometry=PublicGeometry(state)
        self.assertFalse(motion_check(geometry,(4.99,1.5,1.5),(1.5,0,0),(),(4.99,1.5,1.5),SPEC,CONTRACT,.75,0,16)[0])
        self.assertTrue(motion_check(geometry,(1.,1.5,1.5),(0,0,0),(),(1.,1.5,1.5),SPEC,CONTRACT,.75,0,16)[0])
        self.assertFalse(ExecutorStandbyBridge(owner).summary()['safety_guarantee'])

    def test_prediction_uses_executor_gain_and_bounded_number_of_steps(self):
        owner,state,allocation,context=fixture(occupied=False)
        calls=[]
        def step(p,v,target,spec,contract):
            calls.append(contract['prior_strength_search']);return np.asarray(p),np.asarray(v)
        with patch('chapter3_bser.safety.executor_standby.nominal_step',side_effect=step):
            self.assertTrue(motion_check(PublicGeometry(state),state.agents[3].position,(0,0,0),(),(8,1,1),SPEC,CONTRACT,.75)[0])
        self.assertEqual(calls,[CONTRACT['prior_strength_executor']]*24)

    def test_report_separates_collision_from_planner_fallback(self):
        row=dict(scenario_id='scene',actual_length=29,found_step=None,success=False,termination_reason='obstacle_collision',
            first_collision_step=29,first_collision_agent_ids=[3],first_collision_phase='Search')
        r=episode_record('guard_off','NEW_ZERO',row)
        self.assertTrue(r['executor_search_collision']);self.assertFalse(r['guard_available'])
        row['first_collision_phase']='Intercept'
        self.assertFalse(episode_record('guard_off','NEW_ZERO',row)['executor_search_collision'])

    def test_short_probe_and_formal_spec_keep_full_map_kernel(self):
        import json
        from project_paths import ROOT
        spec=json.loads((ROOT/'configs/executor_standby_pilot_v1.json').read_text(encoding='utf-8'))
        validated=validate_spec(spec)
        old=json.loads((ROOT/'configs/coupled_search_bounded_v3.json').read_text(encoding='utf-8'))
        self.assertEqual(validated['new_map_config'],old)
        self.assertEqual(validated['groups'],['OLD_ZERO','NEW_ZERO'])


if __name__=='__main__':unittest.main()
