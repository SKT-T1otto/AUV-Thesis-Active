"""Passive capture, independent geometry, historical binding and negative gates."""
import ast
import copy
import gzip
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
from project_paths import ROOT
from tools import executor_trace as trace
from tools import executor_trace_analysis as analysis
from tools import executor_diagnostics as diag


def fixture(root, *, arm='guard_off'):
    from chapter3_bser.learning.checkpoint import digest
    from chapter3_bser.learning.map_contract import identity
    from chapter3_bser.safety.executor_standby import OPTIONS
    c=trace.read(ROOT/'configs/b2.json')
    c['map_experiment']=dict(schema='auv.active.map_condition.v1',map_id='OLD',research_config=None,
        protocol_sha256='e'*64,executor_safety=dict(OPTIONS,enabled=arm=='guard_on'))
    scene=copy.deepcopy(trace.read(ROOT/'verification/mechanism_scenes.json')['scenarios'][0])
    scene.update(scenario_id='eval_gen827699__unknown_validation_m20_0002',scenario_role='eval',scenario_split='eval',max_steps=400)
    source=dict(review_sha256=trace.read(ROOT/'provenance/executor_standby_manifest.json')['sha256'])
    manifest=dict(scenarios=[scene],scenario_count=1)
    expected=dict(actual_length=96,found_step=None,success=False,termination_reason='obstacle_collision',
        first_collision_step=96,first_collision_phase='Search',first_collision_agent_ids=[3],collision_records=[],
        source_review=source,trajectory_complete=True,scenario_id=scene['scenario_id'],scenario_sha256=digest(scene),map_condition=identity(c))
    output=root/'jobs/OLD_ZERO/evaluation';output.mkdir(parents=True)
    (root/'inputs').mkdir()
    trace.write(root/'inputs/evaluation.json',manifest);trace.write(root/'inputs/OLD_ZERO.json',c)
    (output/'episodes.jsonl').write_text(json.dumps(expected)+'\n',encoding='utf-8')
    receipt=dict(schema='auv.active.evaluation.v1',status='completed',completed=1,source_review=source,
        manifest_sha256=digest(manifest),method='D2',map_condition=identity(c),train_map='OLD',eval_map='OLD',
        files={'episodes.jsonl':trace.sha(output/'episodes.jsonl')})
    expected.update(train_map='OLD',eval_map='OLD')
    (output/'episodes.jsonl').write_text(json.dumps(expected)+'\n',encoding='utf-8')
    receipt['files']['episodes.jsonl']=trace.sha(output/'episodes.jsonl')
    trace.write(output/'evaluation_identity.json',receipt)
    plan=dict(schema='auv.active.map_suite.v1',source_review=source,
        jobs=[dict(name='OLD_ZERO',config='inputs/OLD_ZERO.json',evaluation='jobs/OLD_ZERO/evaluation',map_condition=identity(c))],
        assets={n:trace.sha(root/n) for n in ('inputs/evaluation.json','inputs/OLD_ZERO.json')})
    plan['sha256']=digest(plan);trace.write(root/'plan.json',plan)
    return c,scene,expected


class PassiveRecorder(unittest.TestCase):
    def test_tap_forwards_same_arguments_and_result_once(self):
        argument,answer=object(),object();calls=[];seen=[]
        class Owner:
            def method(self,*args,**kwargs):calls.append((args,kwargs));return answer
        owner=Owner()
        def observer(when,args,kwargs,result):seen.append(when)
        with trace.tap_method(owner,'method',observer):
            self.assertIs(owner.method(argument,key=argument),answer)
        self.assertEqual(len(calls),1);self.assertIs(calls[0][0][0],argument)
        self.assertIs(calls[0][1]['key'],argument);self.assertNotIn('method',vars(owner))
        self.assertEqual(seen,['before','after'])

    def test_tap_restores_instance_override_on_exception(self):
        def original(*args):raise RuntimeError('physics failure')
        owner=NS(method=original)
        with self.assertRaisesRegex(RuntimeError,'physics failure'):
            with trace.tap_method(owner,'method',lambda *args:None):owner.method(1)
        self.assertIs(owner.method,original)

    def test_read_fields_never_call_lazy_property(self):
        class Trap:
            @property
            def known_free_mask(self):raise AssertionError('planner was refreshed')
        self.assertEqual(trace.fields(Trap(),trace.MAP_FIELDS),{})

    def test_snapshot_has_no_planning_queries_or_hidden_geometry(self):
        source=inspect.getsource(trace.snapshot);tree=ast.parse(source)
        names={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(names & {'live','refresh','query','tracking_target','initialize','compile_guidance','extract_planning_state'})
        for name in ('obstacles','_flow','scenario'):self.assertNotIn(name,source)

    def test_taps_do_not_recompute_controller_events(self):
        source=inspect.getsource(trace.Taps)
        self.assertNotIn('.detect(',source);self.assertNotIn('.query(',source)
        self.assertIn("'observe_controller_result'",source)

    def test_blob_roundtrip_tamper_and_existing_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'blobs';blobs=trace.Blobs(root)
            identity=blobs.put(dict(a=torch.tensor([1.,2.])))
            self.assertEqual(blobs.get(identity),dict(a=[1.,2.]))
            self.assertEqual(blobs.put(dict(a=[1.,2.])),identity)
            with self.assertRaises(FileExistsError):trace.Blobs(root)
            (root/(identity+'.json.gz')).write_bytes(gzip.compress(b'{}'))
            with self.assertRaisesRegex(ValueError,'changed'):blobs.get(identity)

    def test_blob_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            blobs=trace.Blobs(Path(folder)/'blobs')
            with self.assertRaises(ValueError):blobs.get('../secret')

    def test_typed_cache_keys_do_not_alias(self):
        self.assertNotEqual(trace.digest({1:'a'}),trace.digest({'1':'a'}))

    def test_summary_excludes_only_wall_timers(self):
        a=dict(compile_seconds=1,counts=dict(checks=2),success=False)
        b=dict(a,compile_seconds=3)
        self.assertEqual(trace.behavioral_summary(a),trace.behavioral_summary(b))
        b['success']=True
        self.assertNotEqual(trace.behavioral_summary(a),trace.behavioral_summary(b))

    def test_modified_recording_cannot_be_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'trace.jsonl').write_text('')
            trace.write(p/'episode.json',dict(actual_length=1))
            trace.write(p/'signatures.json',['0'*64,'1'*64])
            trace.write(p/'recording.json',dict(schema=trace.SCHEMA,completed=True,instrumented=False,physical_steps=1,
                files={f.name:trace.sha(f) for f in p.iterdir()}))
            trace.verified_recording(p)
            (p/'trace.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'inventory changed'):trace.verified_recording(p)


class GeometryAndMotion(unittest.TestCase):
    def test_closed_slab_stationary_surface_and_tangent(self):
        self.assertEqual(analysis.slab([0,0,0],[0,0,0],[0,0,0],[1,1,1]),0)
        self.assertEqual(analysis.slab([-1,0,0],[1,0,0],[0,0,0],[1,1,1]),.5)
        self.assertIsNone(analysis.slab([-1,-.01,0],[1,-.01,0],[0,0,0],[1,1,1]))

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):analysis.slab([0,0,float('nan')],[1,1,1],[0,0,0],[1,1,1])

    def test_clearance_minimum_is_between_endpoints(self):
        self.assertAlmostEqual(analysis.segment_clearance([-1,2,0],[2,2,0],[0,0,-1],[1,1,1]),1)
        self.assertEqual(analysis.segment_clearance([-1,.5,0],[2,.5,0],[0,0,-1],[1,1,1]),0)

    def test_empty_path_is_unavailable(self):
        self.assertFalse(analysis.path_geometry([],[0,0,0],[1,1,1])['available'])
        self.assertIsNone(analysis.polyline_distance([0,0,0],[]))

    def test_polyline_deviation_does_not_prepend_current_position(self):
        self.assertEqual(analysis.polyline_distance([.5,2,0],[[0,0,0],[1,0,0]]),2)

    def test_original_0002_top_and_0006_side_contacts(self):
        examples=[([8.850747108459473,8.51616096496582,5.474787712097168],[8.92465877532959,8.591042518615723,5.422943115234375],
            [9.1238441,10.9130107,4.2407877],[1.7381187,4.7247676,2.367216],.06105125355782127,.9719846),
            ([10.629842758178711,14.175718307495117,3.5335991382598877],[10.430890083312988,14.036452293395996,3.533639430999756],
            [9.6702346,14.1500746,3.9901324],[1.6291286,3.8310229,2.9299561],.14504385817870968,.7290345)]
        for p,q,c,s,d,t in examples:
            lo,hi=np.asarray(c)-np.asarray(s)/2,np.asarray(c)+np.asarray(s)/2
            self.assertAlmostEqual(analysis.point_clearance(p,lo,hi),d)
            self.assertAlmostEqual(analysis.slab(p,q,lo,hi),t,places=5)

    def test_guard_0011_semantic_target_is_far_from_collision(self):
        p=np.array([15.043041229248047,10.850539207458496,4.370895862579346])
        target=np.array([17.226137161254883,18.657411575317383,4.236938953399658])
        self.assertAlmostEqual(float(np.linalg.norm(p-target)),8.107472384775688)

    def test_exact_voxel_oracle_can_disagree_with_old_sampling(self):
        centers=[[.5,.5,0],[.5,1.5,0],[1.5,.5,0],[1.5,1.5,0]]
        occupied=np.array([False,True,False,False]).reshape(2,2,1)
        live=dict(cell_dx=1,cell_dy=1,cell_dz=1,grid_size=[2,2,1],eps=1e-8,z_range=[0,1],_space_size_np=[2,2,1],known_occupied_mask=occupied)
        p,q=[.1,.1,0],[1.9,1.9,0]
        self.assertTrue(analysis.sampled_segment(p,q,live)['accepted'])
        self.assertEqual(analysis.public_segment(p,q,dict(cell_centers=centers,spacing=[1,1,1]),occupied,np.zeros(4,bool))['occupied_cells'],[1])

    def test_unknown_voxel_is_not_known_free(self):
        r=analysis.public_segment([0,0,0],[1,0,0],dict(cell_centers=[[0,0,0],[1,0,0]],spacing=[1,1,1]),[False,False],[True,False])
        self.assertEqual(r['unknown_cells'],[0]);self.assertEqual(r['occupied_cells'],[])

    def test_current_position_target_is_velocity_damping(self):
        control=dict(eps=1e-8,prior_slow_radius_xy=2.4,prior_slow_radius_z=1.2,prior_kv_xy=1.1,prior_kv_z=1.,prior_strength_executor=1.,use_residual_prior=True)
        spec=dict(v_xy_max=1.5,v_z_max=.75,a_xy_max=.8,a_z_max=.55,drag_xy=.16,drag_z=.24,buoyancy_bias=.02)
        a=analysis.nominal_acceleration([1,1,1],[.1,-.2,.01],[1,1,1],spec,control)
        np.testing.assert_allclose(a,[-.11,.22,-.01])
        v=analysis.velocity_step([.1,-.2,.01],a,spec,.2)
        self.assertNotEqual(v.tolist(),[0,0,0]);self.assertGreater(v[2],0)


class AdmissionAndHistory(unittest.TestCase):
    def test_prepared_plan_preserves_frozen_behavior_and_rejects_resigned_config_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as folder:
            root=Path(folder);experiment=root/'experiment';fixture(experiment/'guard_off')
            run=root/'collision_terminal/run'
            diag.prepare(experiment,run,cases=['baseline_0002'])
            plan=diag.load_plan(run);case=plan['cases'][0]
            config=trace.read(run/case['config']);config['rl']['gamma']=.9
            (run/case['config']).write_text(json.dumps(config))
            plan['assets'][case['config']]=trace.sha(run/case['config'])
            plan['sha256']=trace.digest({k:v for k,v in plan.items() if k!='sha256'})
            (run/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'beyond output metadata'):diag.load_plan(run)

    def test_prepare_cannot_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as folder:
            root=Path(folder)/'collision_terminal/run';root.mkdir(parents=True)
            with self.assertRaises(FileExistsError):diag.prepare('missing',root,cases=['baseline_0002'])

    def test_pair_run_and_report_with_synthetic_unit_recordings(self):
        def fake_episode(config,scene,path,*,instrumented):
            # No simulator is constructed; this exercises receipts/report plumbing.
            path.mkdir(parents=True,exist_ok=False)
            summary={k:None for k in diag.EVENT_FIELDS}
            summary.update(actual_length=1,found_step=None,success=False,termination_reason='timeout',
                first_collision_agent_ids=[],collision_records=[])
            trace.write(path/'episode.json',summary);trace.write(path/'signatures.json',['0'*64,'1'*64])
            spec=dict(v_xy_max=1.5,v_z_max=.75,a_xy_max=.8,a_z_max=.55,drag_xy=.16,drag_z=.24,buoyancy_bias=.02)
            control=dict(eps=1e-8,dt=.2,prior_slow_radius_xy=2.4,prior_slow_radius_z=1.2,
                prior_kv_xy=1.1,prior_kv_z=1.,prior_strength_executor=1.,use_residual_prior=True)
            trace.write(path/'public_control.json',dict(specs=[spec]*4,control=control,bounds=[20,20,8]))
            before=dict(_agent_pos=[[1,1,1]]*4,_agent_vel=[[0,0,0]]*4,_nav_targets=[[1,1,1]]*4)
            after=dict(_agent_pos=[[1,1,1.0008]]*4,_agent_acc=[[0,0,0]]*4,
                _last_prior_acc=[[0,0,0]]*4,_last_residual_acc=[[0,0,0]]*4)
            frame=dict(schema=trace.SCHEMA,step_before=0,transition=dict(t=0),before=dict(executor_guidance=dict(tracking_waypoint=[1,1,1])),
                physical_step=dict(before=dict(physical=before,phase='Search'),after=after,collisions=[]))
            (path/'trace.jsonl').write_text(json.dumps(frame)+'\n' if instrumented else '')
            trace.write(path/'recording.json',dict(schema=trace.SCHEMA,completed=True,instrumented=instrumented,
                physical_steps=1,snapshot_checks=int(instrumented),forwarded_physics_calls=int(instrumented),
                files={p.name:trace.sha(p) for p in path.iterdir() if p.is_file()},synthetic_unit_fixture=True))
            return summary,['0'*64,'1'*64]
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as folder:
            root=Path(folder);experiment=root/'experiment';fixture(experiment/'guard_off')
            run=root/'collision_terminal/run';diag.prepare(experiment,run,cases=['baseline_0002'])
            validation=root/'synthetic_validation.json';trace.write(validation,dict(synthetic_unit_fixture=True))
            with patch.object(diag,'run_episode',side_effect=fake_episode),patch('chapter3_bser.learning.map_suite.require_linux_validation'),patch('chapter3_bser.learning.provenance.research_readiness'):
                result=diag.run(run,validation)
            self.assertTrue(result['recording_parity_passed']);self.assertFalse(result['all_original_terminal_events_exact'])
            report=diag.report(run,root/'collision_terminal/report')
            self.assertFalse(report['analyses'][0]['executor_collision'])
            self.assertFalse(report['causal_ranking_established'])
            result['recording_parity_passed']=False
            (run/'run.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'aggregate claims changed'):diag.report(run,root/'collision_terminal/report2')

    def test_recording_clock_mismatch_rejected_even_with_resigned_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);trace.write(p/'episode.json',dict(actual_length=2))
            trace.write(p/'signatures.json',['0'*64,'1'*64]);(p/'trace.jsonl').write_text('')
            trace.write(p/'recording.json',dict(schema=trace.SCHEMA,completed=True,physical_steps=1,instrumented=False,
                files={f.name:trace.sha(f) for f in p.iterdir()}))
            with self.assertRaisesRegex(ValueError,'clock/signature'):trace.verified_recording(p)

    def test_historical_frozen_old_zero_is_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config,scene,expected=fixture(root)
            c,scenes,rows=diag.historical_inputs(root,'guard_off')
            self.assertEqual(c,config);self.assertEqual(scenes,[scene]);self.assertEqual(rows,[expected])

    def test_changed_frozen_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root)
            (root/'inputs/evaluation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'frozen historical input changed'):diag.historical_inputs(root,'guard_off')

    def test_changed_native_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root)
            (root/'jobs/OLD_ZERO/evaluation/episodes.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'rows changed'):diag.historical_inputs(root,'guard_off')

    def test_guard_on_cannot_be_used_as_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root,arm='guard_on')
            with self.assertRaisesRegex(ValueError,'guard condition changed'):diag.historical_inputs(root,'guard_off')

    def test_unsafe_historical_job_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            from chapter3_bser.learning.checkpoint import digest
            root=Path(folder);fixture(root);plan=trace.read(root/'plan.json')
            plan['jobs'][0]['config']='../config.json'
            plan['sha256']=digest({k:v for k,v in plan.items() if k!='sha256'})
            (root/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'unsafe historical'):diag.historical_inputs(root,'guard_off')

    def test_missing_terminal_fields_are_not_exact_matches(self):
        with self.assertRaisesRegex(ValueError,'incomplete native'):diag.event_comparison({}, {})

    def test_phase_transfer_does_not_match_search_collision(self):
        a={k:None for k in diag.EVENT_FIELDS};b=dict(a)
        a['first_collision_phase']='Search';b['first_collision_phase']='Intercept'
        self.assertFalse(diag.event_comparison(a,b)['exact_terminal_event_match'])

    def test_touch_point_difference_is_preserved(self):
        a={k:None for k in diag.EVENT_FIELDS};b=copy.deepcopy(a)
        a['collision_records']=[dict(first_hit_point=[1,2,3])]
        b['collision_records']=[dict(first_hit_point=[1,2,3.000001])]
        self.assertFalse(diag.event_comparison(a,b)['fields']['collision_records'])

    def test_no_execute_never_creates_output_or_runtime(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(diag,'prepare') as prepare:
            path=Path(folder)/'absent'
            result=diag.main(['prepare','--root',str(path),'--experiment','missing'])
            self.assertFalse(result['executed']);prepare.assert_not_called();self.assertFalse(path.exists())

    def test_validation_rejection_precedes_runtime(self):
        with patch.object(diag,'load_plan',return_value=dict(source={})),patch(
             'chapter3_bser.learning.map_suite.require_linux_validation',side_effect=ValueError('stale validation')),patch.object(diag,'run_episode') as runtime:
            with self.assertRaisesRegex(ValueError,'stale validation'):diag.run('unused','unused')
            runtime.assert_not_called()

    def test_existing_or_external_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises((ValueError,FileExistsError)):diag.output_path(folder)
        with self.assertRaises(ValueError):diag.output_path(ROOT.parent/'collision_terminal/external')

    def test_production_behavior_does_not_import_observer(self):
        for path in (ROOT/'core').rglob('*.py'):
            self.assertNotIn('tools.executor_trace',path.read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
