"""Independent admission/integrity checks; no environment or model restore."""
import ast,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from project_paths import ROOT
import checks.test_map_source as source_checks
from tools.reach_avoid_monitor_source import MANIFEST,EVOLUTION,verify,digest


class ReachAvoidRelease(unittest.TestCase):
    def test_control_constants_match_original_source_not_mock_fixture(self):
        from chapter3_bser.reach_avoid.options import CONTRACT
        tree=ast.parse((ROOT/'core/env/uav_env.py').read_text())
        eps=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='eps' for t in n.targets))
        self.assertEqual(CONTRACT['eps'],ast.literal_eval(eps))
        import inspect
        from core.env.uav_env import UAVEnv
        defaults=inspect.signature(UAVEnv.__mro__[1].__init__).parameters
        for key,value in CONTRACT.items():
            if key!='eps':self.assertEqual(value,defaults[key].default)
    def test_case_subset_cannot_repeat_or_invent_native_cases(self):
        from tools.check_reach_avoid import select_cases
        for cases in ([],['rule_0002','rule_0002'],['other'],['composition_0002']):
            with self.assertRaises(ValueError):select_cases(cases)
    def test_non_nominal_branch_preserves_original_task_gate_bookkeeping(self):
        import copy
        from collections import Counter
        from types import SimpleNamespace as NS
        import torch,numpy as np
        from checks.reach_avoid_fixtures import public_controller
        from checks.test_safe_rl_recovery import context
        from chapter3_bser.reach_avoid.runtime import ReachAvoidRuntime
        c,p,s,_=public_controller();runtime=ReachAvoidRuntime.__new__(ReachAvoidRuntime)
        runtime.behavior='rule';runtime.ra=c;runtime._active_exploration=False
        runtime.repair_counts=Counter();runtime._gates=lambda:np.array([True,False,True,False])
        runtime._packet=lambda:p;runtime.guidance=context(s);runtime.env=NS(install_guidance=lambda g:None)
        actions=runtime._actions_for_step(torch.ones((4,3)))
        self.assertEqual(tuple(actions.shape),(4,3));self.assertEqual(runtime.repair_counts['disabled_agent_steps'],2)
        np.testing.assert_array_equal(runtime.applied_gates,[True,False,True,False])
        self.assertTrue(torch.equal(runtime.proposed_actions,torch.ones((4,3))))
    def alter_and_resign(self,root,name,text):
        (root/name).write_text(text,encoding='utf-8')
        h=hashlib.sha256((root/name).read_bytes()).hexdigest()
        evo=json.loads((root/EVOLUTION).read_text());changes={r['path']:r for r in evo['changes']}
        if name not in changes:evo['changes'].append(dict(path=name,before_sha256=h,reason='negative control'))
        next(r for r in evo['changes'] if r['path']==name)['after_sha256']=h
        (root/EVOLUTION).write_text(json.dumps(evo),encoding='utf-8')
        m=json.loads((root/MANIFEST).read_text());m['files'][name]=h
        m['files'][EVOLUTION]=hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
        m['sha256']=digest({k:v for k,v in m.items() if k!='sha256'})
        (root/MANIFEST).write_text(json.dumps(m),encoding='utf-8')
    def test_resigning_cannot_hide_truth_target_access(self):
        with source_checks.source_directory() as folder:
            root,_=source_checks.MapSource().fixture(folder);name='core/env/mission_env.py'
            text=(root/name).read_text().replace("getattr(self._env, 'executor_delivered_target_state', None)","self._env.target_state")
            self.alter_and_resign(root,name,text)
            with self.assertRaisesRegex(ValueError,'read-only API'):verify(root)
    def test_resigning_cannot_hide_handoff_side_effect(self):
        with source_checks.source_directory() as folder:
            root,_=source_checks.MapSource().fixture(folder);name='core/env/mission_env.py'
            text=(root/name).read_text().replace("if not bool(self._env.executor_target_assigned):","self._env.step_count += 1\n        if not bool(self._env.executor_target_assigned):")
            self.alter_and_resign(root,name,text)
            with self.assertRaisesRegex(ValueError,'read-only API'):verify(root)
    def test_new_runtime_cannot_import_verification_tools(self):
        with source_checks.source_directory() as folder:
            root,_=source_checks.MapSource().fixture(folder);name='chapter3_bser/reach_avoid/runtime.py'
            self.alter_and_resign(root,name,(root/name).read_text()+'\nimport tools.check_reach_avoid\n')
            with self.assertRaisesRegex(ValueError,'tool imported by runtime'):verify(root)
    def test_mechanism_plan_respects_shared_eight_case_limit(self):
        from tools.check_reach_avoid import CASES
        self.assertEqual(len(CASES),8);self.assertEqual(len({x[0] for x in CASES}),8)
        self.assertEqual(sum(x[2]=='learned' for x in CASES),1)
    def test_flow_fixture_cannot_switch_to_capture_before_32_steps(self):
        import numpy as np
        from tools.check_reach_avoid import custom_scene
        from tools.audit_safe_speed import declared_specs
        scene=custom_scene({},'flow');target=np.array(scene['target_initial_position'])
        motion=np.linalg.norm(scene['target_initial_velocity'][:2])*32*.2
        for i,spec in enumerate(declared_specs()[:3]):
            remaining=np.linalg.norm(target[:2]-scene['initial_agent_positions'][i][:2])-spec['v_xy_max']*32*.2-motion
            self.assertGreater(remaining,spec['sensor_range'])
    def test_new_mechanism_entrypoint_is_dry_by_default(self):
        from tools.check_reach_avoid import main
        with patch('tools.check_reach_avoid.run',side_effect=AssertionError('must not execute')):
            main(['--output','missing'])
    def test_fault_fixture_only_supplies_goal_points_to_runtime(self):
        tree=ast.parse((ROOT/'tools/check_reach_avoid.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ReachAvoidRuntime')
        keywords={k.arg for k in call.keywords}
        self.assertIn('goal_faults',keywords);self.assertNotIn('true_obstacles',keywords)
