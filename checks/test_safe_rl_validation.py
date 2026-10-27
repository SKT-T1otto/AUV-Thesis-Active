"""Host-scoped parity and diagnostic-only archive tests; zero environment starts."""
import copy,hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from tools.safe_rl_validation import signature_comparison,require_same_host_repeat,numerical_frames,archived_report

def frames(count=2):
    return [dict(t=i,positions=[[float(i),1.,2.]]*4,velocities=[[.1,0.,0.]]*4,
        rewards=[.01]*4,tracking=[[5.,5.,4.]]*4) for i in range(count)]

class SafeRLValidation(unittest.TestCase):
    def test_same_host_full_signature_exact_required(self):
        a=['a'*64,'b'*64,'c'*64]
        self.assertTrue(require_same_host_repeat(a,list(a),physical_steps=2)['passed'])
    def test_same_host_rng_map_or_state_change_rejected(self):
        with self.assertRaisesRegex(ValueError,'same-host baseline repeat differs'):
            require_same_host_repeat(['a'*64,'b'*64],['a'*64,'c'*64],physical_steps=1)
    def test_truncated_repeat_rejected(self):
        with self.assertRaisesRegex(ValueError,'same-host'):
            require_same_host_repeat(['a'*64,'b'*64],['a'*64],physical_steps=1)
    def test_missing_initial_signature_rejected(self):
        with self.assertRaisesRegex(ValueError,'same-host'):
            require_same_host_repeat(['a'*64],['a'*64],physical_steps=1)
    def test_invalid_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid complete'):
            signature_comparison(['a'],['a'])
    def test_unbounded_repeat_rejected(self):
        with self.assertRaisesRegex(ValueError,'bounded'):
            require_same_host_repeat(['a'*64]*34,['a'*64]*34,physical_steps=33)
    def test_empty_digests_do_not_establish_identity(self):
        self.assertFalse(signature_comparison([],[])['exact'])
    def test_reference_difference_report_retains_first_mismatch(self):
        r=signature_comparison(['a'*64,'b'*64],['a'*64,'c'*64,'d'*64])
        self.assertFalse(r['exact']);self.assertEqual(r['first_mismatch_index'],1);self.assertEqual(r['mismatch_count'],2)
    def test_numerical_difference_is_quantified_without_rounding_hash(self):
        a=frames();b=copy.deepcopy(a);b[0]['positions'][0][0]+=1e-7
        r=numerical_frames(a,b)
        self.assertTrue(r['numerically_close_at_5e_5']);self.assertAlmostEqual(r['max_absolute_errors']['positions'],1e-7)
    def test_large_reference_difference_is_reported_as_not_close(self):
        a=frames();b=copy.deepcopy(a);b[0]['tracking'][0][0]+=.1
        r=numerical_frames(a,b);self.assertFalse(r['numerically_close_at_5e_5']);self.assertAlmostEqual(r['max_absolute_errors']['tracking'],.1)
    def test_nonfinite_reference_is_invalid(self):
        a=frames();a[0]['rewards'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):numerical_frames(a,frames())
    def test_shape_mismatch_cannot_look_numerically_close(self):
        a=frames();a[0]['positions']=[[1.,2.,3.]]
        r=numerical_frames(a,frames());self.assertFalse(r['complete']);self.assertFalse(r['numerically_close_at_5e_5'])
    def test_frame_clock_mismatch_rejected(self):
        a=frames();a[0]['t']=1
        with self.assertRaisesRegex(ValueError,'clock'):numerical_frames(a,frames())
    def test_cross_host_archive_mismatch_does_not_claim_exact_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'archive.zip'
            with zipfile.ZipFile(p,'x') as z:
                z.writestr('baseline_0002/signatures.json',json.dumps(['b'*64]*3))
                z.writestr('baseline_0002/physical_frames.json',json.dumps(frames()))
            r=archived_report(['a'*64]*3,frames(),p,dict(platform='other-host'))
            self.assertFalse(r['admission_gate']);self.assertFalse(r['signatures']['exact'])
            self.assertTrue(r['physical']['numerically_close_at_5e_5'])
            self.assertFalse(r['cross_platform_bitwise_reproducibility_established'])
    def test_resigning_manifest_cannot_change_unrepaired_safety_learner(self):
        from checks import test_map_source as sources
        from tools.safe_rl_transport_source import MANIFEST,EVOLUTION,verify,digest
        with sources.source_directory() as directory:
            root,manifest=sources.MapSource().fixture(directory)
            name='chapter3_bser/safe_rl/learner.py';p=root/name;p.write_bytes(p.read_bytes()+b'\n# invalid verification-only algorithm edit\n')
            h=hashlib.sha256(p.read_bytes()).hexdigest();e=json.loads((root/EVOLUTION).read_text(encoding='utf-8'))
            e['changes'].append(dict(path=name,before_sha256=manifest['files'][name],after_sha256=h,reason='negative control'))
            (root/EVOLUTION).write_text(json.dumps(e),encoding='utf-8');manifest['files'][name]=h
            manifest['files'][EVOLUTION]=hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
            manifest['sha256']=digest({k:v for k,v in manifest.items() if k!='sha256'})
            (root/MANIFEST).write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'protected parent runtime changed'):verify(root)
