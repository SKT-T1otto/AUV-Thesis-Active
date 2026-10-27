"""Transport/prefix regression only; no native environment or checkpoint restore."""
import contextlib,copy,hashlib,io,json,os,platform,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from project_paths import ROOT
from tools.check_process import run_checks_process
from tools.safe_rl_resume import load_prefix,claim_prefix,require_prefix_unchanged,PREFIX,PARENT_REVIEW,PARENT_COMMIT

ARCHIVE=ROOT/'verification/safe_rl_recovery_v2'/PARENT_REVIEW/'native_evidence.zip'


def write(p,data):p.write_text(json.dumps(data),encoding='utf-8')


@contextlib.contextmanager
def prefix_fixture():
    # Real published Windows trace DATA inside a synthetic host/timeout wrapper.
    # It exercises parser/transport logic and is never Linux execution evidence.
    with tempfile.TemporaryDirectory(dir=ROOT/'work') as folder:
        root=Path(folder);native=root/'checks/safe_rl';native.mkdir(parents=True)
        allowed={name for name,_ in PREFIX}|{f'initialization_{i:02d}.json' for i in range(1,6)}
        with zipfile.ZipFile(ARCHIVE) as z:
            for name in z.namelist():
                if name.split('/')[0] not in allowed:continue
                target=native/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
        source=json.loads((native/'initialization_01.json').read_text())['source']
        write(root/'validation.json',dict(passed=False,traceback='TimeoutExpired synthetic transport fixture',source=source,
            git=dict(head=PARENT_COMMIT,clean=True),package_before=dict(passed=True),platform=platform.platform(),python=sys.version,
            linux_execution=platform.system()=='Linux',versions=dict(numpy=np.__version__,torch=torch.__version__),synthetic_fixture=True))
        write(root/'checks/structural.json',dict(passed=True,source=dict(manifest_sha256=PARENT_REVIEW)))
        write(root/'checks/regression.json',dict(passed=True,tests_run=363,expected=363,failures=[],errors=[],skipped=[]))
        yield root


class PrefixValidation(unittest.TestCase):
    def test_complete_prefix_retains_parent_identity_and_zero_restore(self):
        with prefix_fixture() as root:
            r=load_prefix(root);self.assertEqual(r['receipt']['environment_initializations'],5)
            self.assertEqual(r['receipt']['physical_steps'],160);self.assertFalse(r['receipt']['results_relabelled'])
            self.assertTrue(all(x['source_identity']['review_sha256']==PARENT_REVIEW for x in r['records']))
    def test_reading_and_rechecking_never_changes_parent_files(self):
        with prefix_fixture() as root:
            r=load_prefix(root);before=dict(r['receipt']['files']);require_prefix_unchanged(r['receipt'])
            self.assertEqual(before,load_prefix(root)['receipt']['files'])
    def test_sixth_reservation_prevents_budget_reset(self):
        with prefix_fixture() as root:
            write(root/'checks/safe_rl/initialization_06.json',{})
            with self.assertRaisesRegex(ValueError,'exactly five'):load_prefix(root)
    def test_partial_fifth_case_is_rejected(self):
        with prefix_fixture() as root:
            p=root/'checks/safe_rl/rule_0006/episode.json';r=json.loads(p.read_text());r['actual_length']=31;write(p,r)
            with self.assertRaisesRegex(ValueError,'incomplete'):load_prefix(root)
    def test_collided_fifth_case_is_rejected(self):
        with prefix_fixture() as root:
            p=root/'checks/safe_rl/rule_0006/episode.json';r=json.loads(p.read_text());r['first_collision_agent_ids']=[3];write(p,r)
            with self.assertRaisesRegex(ValueError,'collided'):load_prefix(root)
    def test_altered_baseline_repeat_is_rejected(self):
        with prefix_fixture() as root:
            p=root/'checks/safe_rl/baseline_repeat_0002/signatures.json';r=json.loads(p.read_text());r[1]='f'*64;write(p,r)
            with self.assertRaisesRegex(ValueError,'same-host'):load_prefix(root)
    def test_host_change_is_rejected(self):
        with prefix_fixture() as root:
            p=root/'validation.json';r=json.loads(p.read_text());r['platform']='other host';write(p,r)
            with self.assertRaisesRegex(ValueError,'host/runtime'):load_prefix(root)
    def test_source_identity_cannot_be_relabelled(self):
        with prefix_fixture() as root:
            p=root/'validation.json';r=json.loads(p.read_text());r['source']['inventory_sha256']='0'*64;write(p,r)
            with self.assertRaisesRegex(ValueError,'frozen parent'):load_prefix(root)
    def test_missing_final_acceptance_can_be_recomputed_without_runtime(self):
        with prefix_fixture() as root:
            (root/'checks/safe_rl/rule_0006/acceptance.json').unlink()
            self.assertEqual(load_prefix(root)['receipt']['completed_cases'],5)
    def test_attempted_suffix_directory_is_rejected(self):
        with prefix_fixture() as root:
            (root/'checks/safe_rl/fault_all_four').mkdir()
            with self.assertRaisesRegex(ValueError,'suffix already attempted'):load_prefix(root)
    def test_second_continuation_claim_is_rejected(self):
        with prefix_fixture() as root:
            receipt=load_prefix(root)['receipt'];claim_prefix(receipt,root/'run1',root=root)
            with self.assertRaisesRegex(ValueError,'already claimed'):claim_prefix(receipt,root/'run2',root=root)
    def test_mutation_during_suffix_is_rejected(self):
        with prefix_fixture() as root:
            receipt=load_prefix(root)['receipt'];(root/'checks/safe_rl/rule_0006/episode.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed during'):require_prefix_unchanged(receipt)
    def test_parent_regression_failure_is_not_silently_accepted(self):
        with prefix_fixture() as root:
            p=root/'checks/regression.json';r=json.loads(p.read_text());r['failures']=['bad'];write(p,r)
            with self.assertRaisesRegex(ValueError,'regression'):load_prefix(root)


class ProcessTransport(unittest.TestCase):
    def run_child(self,code,output,**kw):
        env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
        with contextlib.redirect_stdout(io.StringIO()) as stdout,contextlib.redirect_stderr(io.StringIO()) as stderr:
            r=run_checks_process([sys.executable,'-u','-c',code],cwd=ROOT,env=env,output=output,**kw)
        return r,stdout.getvalue(),stderr.getvalue()
    def test_stdout_stderr_are_streamed_and_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            r,out,err=self.run_child("import sys;print('visible');print('warning',file=sys.stderr)",folder)
            self.assertEqual(r['exit_code'],0);self.assertIn('visible',out);self.assertIn('warning',err)
            self.assertIn(b'visible',(Path(folder)/'checks.stdout.log').read_bytes())
            self.assertIn(b'warning',(Path(folder)/'checks.stderr.log').read_bytes())
    def test_timeout_keeps_partial_logs_and_joins_child(self):
        with tempfile.TemporaryDirectory() as folder:
            r,out,err=self.run_child("import time;print('partial',flush=True);time.sleep(10)",folder,timeout=1,heartbeat=.2)
            self.assertTrue(r['timed_out']);self.assertNotEqual(r['exit_code'],0)
            self.assertIn(b'partial',(Path(folder)/'checks.stdout.log').read_bytes());self.assertIn('checks running',out)
    def test_failed_child_exit_is_reported_with_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            r,out,err=self.run_child("import sys;print('failure');sys.exit(9)",folder)
            self.assertEqual(r['exit_code'],9);self.assertFalse(r['timed_out']);self.assertIn('failure',out)
    def test_existing_log_is_not_overwritten_or_child_started(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'checks.stdout.log';p.write_bytes(b'preserve')
            with patch('tools.check_process.subprocess.Popen',side_effect=AssertionError('must not start')):
                with self.assertRaises(FileExistsError):self.run_child('pass',folder)
            self.assertEqual(p.read_bytes(),b'preserve')
    def test_invalid_timeout_does_not_start_child(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'timeout'):self.run_child('pass',folder,timeout=0)


class ContinuationPipeline(unittest.TestCase):
    def test_only_two_suffix_factories_called_parent_case_sources_retained(self):
        from tools import check_safe_rl
        from checks.test_safe_rl import state
        from checks.test_safe_rl_recovery import context
        from tools.safe_rl_resume import claim_prefix as real_claim
        with prefix_fixture() as root:
            calls=[]
            class RecordedRuntime:
                def __init__(self,config,scene,**kwargs):
                    name='buffer_exit_all_four' if scene['scenario_id']=='buffer_exit_four_roles_v2' else 'fault_all_four';calls.append(name)
                    with zipfile.ZipFile(ARCHIVE) as z:
                        self.frames=json.loads(z.read(name+'/physical_frames.json'));self.episode=json.loads(z.read(name+'/episode.json'))
                    self.step=0;self.terminal=False;self.safety_bridge=NS(inner=NS())
                    st=state();trace=self.episode['safety_rl']['trace']
                    st.agents=tuple(NS(agent_id=i,position=tuple(next(r['physical_position'] for r in trace if r['step']==0 and r['agent_id']==i)),velocity=(0.,0.,0.)) for i in range(4))
                    self.initial=tuple(a.position for a in st.agents);self.guidance=context(st)
                    self.env=NS(get_agent_state=self.public,install_guidance=lambda g:None,
                        refresh_observation_after_guidance=lambda:np.zeros((4,28)),unwrapped=NS(_agent_acc=torch.zeros(4,3)))
                def public(self):
                    if not self.step:return NS(positions=self.initial,velocities=((0.,0.,0.),)*4)
                    f=self.frames[self.step-1];return NS(positions=f['positions'],velocities=f['velocities'])
                def advance(self,*args,**kwargs):
                    t=self.step;rows=[r for r in self.episode['safety_rl']['trace'] if r['step']==t]
                    self.env.unwrapped._agent_acc=torch.tensor([r['executed_acceleration'] for r in rows]);self.step+=1;self.terminal=self.step==32
                    return dict(t=t,rewards=np.array(self.frames[t]['rewards']),safety_features=np.zeros((4,192)),safety_rows=rows)
                def summary(self):return copy.deepcopy(self.episode)
                def close(self):pass
            scenes=[dict(scenario_id=f'unknown_validation_m20_{i+1:04d}',scenario_seed=827699+i,
                obstacles=[dict(center=[9.,9.,4.],size=[2.,2.,2.])]) for i in range(12)]
            with patch.object(check_safe_rl,'SafetyRuntime',RecordedRuntime),\
                 patch.object(check_safe_rl,'RepairedMissionRuntime',side_effect=AssertionError('prefix must not rerun')),\
                 patch.object(check_safe_rl,'build_scenario_manifests',return_value={'M20_MOVING_UNKNOWN_MULTI':dict(scenarios=scenes)}),\
                 patch('tools.safe_rl_transport_source.verify',return_value=dict(passed=True,synthetic_parent_transport_fixture=True)),\
                 patch('tools.safe_rl_resume.claim_prefix',side_effect=lambda r,o:real_claim(r,o,root=root)),\
                 contextlib.redirect_stdout(io.StringIO()):
                result=check_safe_rl.run(root/'continuation',prefix=root)
            self.assertTrue(result['passed'],result.get('traceback'));self.assertEqual(calls,['fault_all_four','buffer_exit_all_four'])
            self.assertEqual(result['environment_initializations'],7);self.assertEqual(result['additional_environment_initializations'],2)
            self.assertEqual(result['physical_steps'],224);self.assertEqual(result['additional_physical_steps'],64)
            self.assertTrue(all(r['source_identity']['review_sha256']==PARENT_REVIEW for r in result['records'][:5]))
            self.assertTrue(all(r['source_identity']==result['source'] for r in result['records'][5:]))
