"""Regression of audit findings1/2/5; no native environment construction."""
import copy,unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from checks.test_safe_rl import state,guidance,SPECS,CONTRACT
from chapter3_bser.safe_rl.options import validate
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.evidence import EvidenceGeometry,ScanEvidence
from chapter3_bser.safe_rl.buffer_recovery import ExitCorridor,buffer_exit
from chapter3_bser.safe_rl.dynamics import integrate
from chapter3_bser.safe_rl.navigation import SafetyNavigationBridge
from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.integration.control_context import AgentAssignmentContextV1,ExecutorAssignmentContextV1,BSERControlContextV1
from tools.safe_rl_acceptance import require_recoverable_case,near_buffer_scene


def geometry(points=(),occupied=()):
    e=ScanEvidence()
    for i,p in enumerate(points):e.hits[(i,0,0)]=(np.array(p,float),0)
    return EvidenceGeometry(state(occupied=occupied),e,validate())


class MarginExit(unittest.TestCase):
    def test_zero_cap_still_allows_checked_outward_acceleration(self):
        c=SafetyController(SPECS,CONTRACT,validate());c.evidence.hits[(0,0,0)]=(np.array([6.001,3,4]),0)
        _,_,rows,_=c.act(state(),guidance(),np.zeros((4,3)),())
        r=rows[3];self.assertEqual(r['speed_cap_xy'],0);self.assertEqual(r['mode'],'BUFFER_EXIT');self.assertTrue(r['model_checked'])
        for flow in ((-.18,-.18),(-.18,.18),(.18,-.18),(.18,.18)):
            p,_=integrate([6,3,4],[0,0,0],r['executed_acceleration'],SPECS[3],.2,flow)
            self.assertLess(p[0],6)
    def test_all_vehicle_roles_can_exit_near_hit(self):
        for spec in SPECS:
            r=buffer_exit([6,3,4],[0,0,0],spec,geometry([[6.001,3,4]]),CONTRACT,validate(),[0,0])
            self.assertIsNotNone(r);self.assertLess(r[0][0],0);self.assertEqual(len(r[1]['probes']),5)
    def test_ordinary_segment_check_is_not_relaxed(self):
        g=geometry([[6.001,3,4]])
        self.assertFalse(g.segment([6,3,4],[5.9,3,4]));self.assertTrue(ExitCorridor(g,[6,3,4]).segment([6,3,4],[5.9,3,4]))
    def test_exit_corridor_forbids_approach(self):
        corridor=ExitCorridor(geometry([[6.001,3,4]]),[6,3,4])
        self.assertFalse(corridor.segment([6,3,4],[6.0001,3,4]))
    def test_opposing_evidence_has_no_certified_exit(self):
        self.assertIsNone(buffer_exit([6,3,4],[0,0,0],SPECS[3],geometry([[6.001,3,4],[5.999,3,4]]),CONTRACT,validate(),[0,0]))
    def test_contact_does_not_invent_surface_normal(self):
        self.assertIsNone(buffer_exit([6,3,4],[0,0,0],SPECS[3],geometry([[6,3,4]]),CONTRACT,validate(),[0,0]))
    def test_empty_evidence_does_not_bypass_speed_governor(self):
        self.assertIsNone(buffer_exit([6,3,4],[0,0,0],SPECS[3],geometry(),CONTRACT,validate(),[0,0]))
    def test_inside_coarse_cell_without_hit_remains_unverified(self):
        index=np.ravel_multi_index((3,1,4),(10,10,8))
        self.assertFalse(ExitCorridor(geometry(occupied=[index]),[6.1,3,4.1]).valid)
    def test_exit_cannot_cross_a_different_occupied_cell(self):
        index=np.ravel_multi_index((2,1,4),(10,10,8))
        c=ExitCorridor(geometry([[8.001,3,4.5]],occupied=[index]),[8,3,4.5])
        self.assertFalse(c.segment([8,3,4.5],[4,3,4.5]))
    def test_high_inward_speed_is_not_mislabeled_as_exit(self):
        self.assertIsNone(buffer_exit([6,3,4],[2,0,0],SPECS[3],geometry([[6.001,3,4]]),CONTRACT,validate(),[0,0]))
    def test_padded_voxel_corner_is_classified_as_initial_buffer(self):
        indices=[np.ravel_multi_index((2,1,z),(10,10,8)) for z in (3,4,5)]
        g=geometry([[4.0005,3,4.5]],occupied=indices)
        r=buffer_exit([3.8,3,4.5],[-.34,0,0],SPECS[3],g,CONTRACT,validate(),[.12,0])
        self.assertIsNotNone(r)
    def test_repeated_exit_does_not_stall_on_nominal_margin_edge(self):
        indices=[np.ravel_multi_index((2,1,z),(10,10,8)) for z in (3,4,5)]
        g=geometry([[4.0005,3,4.5]],occupied=indices);p=np.array([3.999,3.,4.5]);v=np.zeros(3)
        for t in range(16):
            flow=.18*np.array([np.sin(.25*p[1]),np.cos(.22*p[0])])
            r=buffer_exit(p,v,SPECS[3],g,CONTRACT,validate(),[0,0] if t==0 else flow)
            if g.distance(p)>.62:break
            self.assertIsNotNone(r);p,v=integrate(p,v,r[0],SPECS[3],.2,flow)
        self.assertGreater(g.distance(p),.62)


def context(st):
    items=tuple(AgentAssignmentContextV1(i,'executor' if i==3 else 'searcher','search','unit',
        (9.,3.,4.),(a.position,(9.,3.,4.)),(9.,3.,4.),a.position,False,True,False) for i,a in enumerate(st.agents))
    ex=ExecutorAssignmentContextV1(3,'unit',(9.,3.,4.),items[3].planned_path,items[3].tracking_waypoint,items[3].hold_position,False,True,False)
    return BSERControlContextV1('bser.control_context.v1','unit','unit',st.step,'SEARCH',items,ex,'unit')


class StagingLifecycle(unittest.TestCase):
    def bridge(self,step=100):
        st=state();st.step=step;ctx=context(st)
        env=NS(read_public_events=lambda cursor:(),get_search_execution_state=lambda:NS(agent_finished=(False,)*4))
        inner=NS(path_tracker=PathTracker(),compile_guidance=lambda *a,**kw:ctx)
        c=SafetyController(SPECS,CONTRACT,validate());b=SafetyNavigationBridge(inner,c,env,validate())
        b.routes[3]=dict(original_goal=(9.,3.,4.),goal=(6.,3.,4.),path=((4.,3.,4.),(6.,3.,4.)),created=0)
        b.quarantine[(3,(9.,3.,4.))]=40
        return b,st,ctx
    def test_arrival_requeries_original_even_when_blocked_zero(self):
        b,st,ctx=self.bridge()
        with patch('chapter3_bser.safe_rl.navigation.extract_planning_state',return_value=st),patch.object(b,'_route',return_value=((6.,3.,4.),(9.,3.,4.))) as query:
            result=b.compile_guidance(None,st,None)
        self.assertEqual(query.call_args.args[:2],(3,(9.,3.,4.)))
        self.assertEqual(result.agent_assignments[3].final_waypoint,(9.,3.,4.));self.assertEqual(b.counts['stage_arrivals'],1)
    def test_retry_survives_query_throttling_and_latches_wait_anchor(self):
        b,st,ctx=self.bridge();b.last_query[3]=99
        with patch('chapter3_bser.safe_rl.navigation.extract_planning_state',return_value=st),patch.object(b,'_route') as query:
            result=b.compile_guidance(None,st,None)
        query.assert_not_called();self.assertIn(3,b.retry_pending);self.assertTrue(result.agent_assignments[3].hold_state)
        self.assertEqual(result.agent_assignments[3].hold_position,(6.,3.,4.))
    def test_expired_stage_requeries_before_arrival(self):
        b,st,ctx=self.bridge();b.routes[3]['goal']=(7.,3.,4.)
        with patch('chapter3_bser.safe_rl.navigation.extract_planning_state',return_value=st),patch.object(b,'_route',return_value=((6.,3.,4.),(9.,3.,4.))):
            b.compile_guidance(None,st,None)
        self.assertEqual(b.counts['stage_expirations'],1);self.assertEqual(b.routes[3]['goal'],(9.,3.,4.))
    def test_phase_change_drops_stale_stage(self):
        b,st,ctx=self.bridge();b.phases[3]=(False,False);st.target_found=True
        with patch('chapter3_bser.safe_rl.navigation.extract_planning_state',return_value=st):b.compile_guidance(None,st,None)
        self.assertNotIn(3,b.routes);self.assertNotIn(3,b.retry_pending)
    def test_repair_unavailable_keeps_retry_pending(self):
        b,st,ctx=self.bridge()
        with patch('chapter3_bser.safe_rl.navigation.extract_planning_state',return_value=st),patch.object(b,'_route',return_value=None):
            result=b.compile_guidance(None,st,None)
        self.assertIn(3,b.retry_pending);self.assertTrue(result.agent_assignments[3].hold_state)


def accepted_fixture():
    initial=np.array([[3.,3.,4.]]*4)
    frames=[dict(positions=(initial+[-(t+1)*.02,0,0]).tolist()) for t in range(32)]
    trace=[dict(step=t,agent_id=i,mode='BUFFER_EXIT',model_checked=True,executed_acceleration=[-.8,0,0]) for t in range(32) for i in range(4)]
    return dict(termination_reason='timeout',safety_rl=dict(trace=trace)),frames,initial


class CollisionAcceptance(unittest.TestCase):
    def test_recoverable_fixture_passes_motion_and_all_role_checks(self):
        s,f,p=accepted_fixture();self.assertTrue(require_recoverable_case(s,f,p,near_buffer=True)['passed'])
    def test_terminal_collision_is_always_rejected(self):
        s,f,p=accepted_fixture();s['termination_reason']='obstacle_collision'
        with self.assertRaisesRegex(ValueError,'collided'):require_recoverable_case(s,f,p)
    def test_collision_record_cannot_hide_under_timeout(self):
        s,f,p=accepted_fixture();s['collision_records']=[dict(agent_id=3)]
        with self.assertRaisesRegex(ValueError,'collided'):require_recoverable_case(s,f,p)
    def test_motionless_timeout_is_not_recovery(self):
        s,f,p=accepted_fixture();f=[dict(positions=p.tolist())]*32
        with self.assertRaisesRegex(ValueError,'stalled'):require_recoverable_case(s,f,p)
    def test_incomplete_role_trace_is_rejected(self):
        s,f,p=accepted_fixture();s['safety_rl']['trace'].pop()
        with self.assertRaisesRegex(ValueError,'incomplete'):require_recoverable_case(s,f,p)
    def test_false_model_checked_label_is_rejected(self):
        s,f,p=accepted_fixture();s['safety_rl']['trace'][0]['mode']='UNCERTIFIED_BRAKE'
        with self.assertRaisesRegex(ValueError,'mislabeled'):require_recoverable_case(s,f,p)
    def test_partial_exit_inside_margin_is_not_accepted(self):
        s,f,p=accepted_fixture();f=[dict(positions=(p+[-(t+1)*.006,0,0]).tolist()) for t in range(32)]
        with self.assertRaisesRegex(ValueError,'inside the original margin'):require_recoverable_case(s,f,p,near_buffer=True)
    def test_near_fixture_replaces_generated_identity_and_geometry(self):
        original=dict(scenario_id='frozen',target_trajectory_sha256='old',target_trajectory_distance=9.)
        s=near_buffer_scene(original)
        self.assertEqual(original['scenario_id'],'frozen');self.assertNotIn('target_trajectory_sha256',s)
        for p,box in zip(s['initial_agent_positions'],s['obstacles']):self.assertAlmostEqual(box['center'][0]-.5-p[0],.0015)
    def test_unrepaired_learning_entrypoints_stop_before_admission(self):
        from chapter3_bser.safe_rl.workflow import execute
        for command,mode in [('train','learned'),('evaluate','learned')]:
            with patch('chapter3_bser.safe_rl.workflow.admission',side_effect=AssertionError('must stop before admission')):
                with self.assertRaisesRegex(ValueError,'3/4/6'):execute(NS(command=command,mode=mode,policy=None))
