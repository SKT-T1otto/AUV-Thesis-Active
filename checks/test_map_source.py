"""Independent evolution negative controls; no runtime or archived-code imports."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from project_paths import ROOT
from tools.safe_rl_recovery_source import verify, digest, BASELINE, EVOLUTION, MANIFEST


def source_directory():
    parent=str(Path(tempfile.gettempdir()).resolve())
    if os.name=='nt' and not parent.startswith('\\\\?\\'):parent='\\\\?\\'+parent
    return tempfile.TemporaryDirectory(dir=parent)


class MapSource(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory).resolve()
        if os.name=='nt' and not str(root).startswith('\\\\?\\'):
            root=Path('\\\\?\\'+str(root))
        manifest = json.loads((ROOT/MANIFEST).read_text(encoding='utf-8'))
        baseline = json.loads((ROOT/BASELINE/'baseline.json').read_text(encoding='utf-8'))
        names = set(manifest['files']) | {MANIFEST, 'provenance/active_manifest.json'}
        names.update(name for name in baseline['files'] if name.startswith('verification/'))
        for name in names:
            target = root/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        return root, manifest

    def test_missing_new_module_rejected(self):
        with source_directory() as directory:
            root, manifest = self.fixture(directory)
            (root/'chapter3_bser/experiments/coupled_search_v1/bounded_belief.py').unlink()
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'): verify(root)

    def test_parent_archive_and_old_evidence_are_immutable(self):
        with source_directory() as directory:
            root, manifest = self.fixture(directory)
            evidence = root/'verification/rebuild/receipt.json'
            evidence.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'parent evidence changed'): verify(root)

    def test_resigned_manifest_cannot_hide_a_physics_change(self):
        with source_directory() as directory:
            root, manifest = self.fixture(directory)
            name = 'core/env/uav_env.py'; source = (root/name).read_text(encoding='utf-8')
            self.assertIn('self._agent_pos += self._agent_vel * self.dt',source)
            (root/name).write_text(source.replace('self._agent_pos += self._agent_vel * self.dt',
                                                'self._agent_pos += self._agent_vel * self.dt * 0.9'),encoding='utf-8')
            changed_hash = hashlib.sha256((root/name).read_bytes()).hexdigest()
            evolution = json.loads((root/EVOLUTION).read_text(encoding='utf-8'))
            row=next((row for row in evolution['changes'] if row['path']==name),None)
            if row is None:
                row=dict(path=name,before_sha256=manifest['files'][name],reason='negative control')
                evolution['changes'].append(row)
            row['after_sha256']=changed_hash
            (root/EVOLUTION).write_text(json.dumps(evolution),encoding='utf-8')
            manifest['files'][name] = changed_hash
            manifest['files'][EVOLUTION] = hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
            manifest['sha256'] = digest({k:v for k,v in manifest.items() if k != 'sha256'})
            (root/MANIFEST).write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'legacy AST changed'): verify(root)

    def test_resigned_manifest_cannot_hide_a_geometry_kernel_change(self):
        with source_directory() as directory:
            root, manifest = self.fixture(directory)
            name='chapter3_bser/experiments/coupled_search_v1/geometry_hypotheses.py'
            path=root/name;source=path.read_text(encoding='utf-8')
            before,entry=source.split('    def entry(self,box):',1)
            source=before+'    def entry(self,box):'+entry.replace('enter=np.maximum(0.,near.max(axis=1));leave=far.min(axis=1)',
                                  'enter=np.maximum(0.,near.max(axis=1))+0.01;leave=far.min(axis=1)')
            path.write_text(source,encoding='utf-8')
            changed_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            evolution=json.loads((root/EVOLUTION).read_text(encoding='utf-8'))
            row=next((r for r in evolution['changes'] if r['path']==name),None)
            if row is None:
                row=dict(path=name,before_sha256=manifest['files'][name],reason='negative control')
                evolution['changes'].append(row)
            row['after_sha256']=changed_hash
            (root/EVOLUTION).write_text(json.dumps(evolution),encoding='utf-8')
            manifest['files'][name]=changed_hash
            manifest['files'][EVOLUTION]=hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
            manifest['sha256']=digest({key:value for key,value in manifest.items() if key!='sha256'})
            (root/MANIFEST).write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'protected geometry kernel changed'):verify(root)

    def test_observer_removal_cannot_hide_a_posterior_change(self):
        with source_directory() as directory:
            root,manifest=self.fixture(directory)
            name='chapter3_bser/experiments/coupled_search_v1/bounded_belief.py'
            path=root/name;source=path.read_text(encoding='utf-8')
            self.assertIn('raw=self.omega*likelihood',source)
            path.write_text(source.replace('raw=self.omega*likelihood','raw=self.omega*likelihood+0.01'),encoding='utf-8')
            changed_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            evolution=json.loads((root/EVOLUTION).read_text(encoding='utf-8'))
            evolution['changes'].append(dict(path=name,before_sha256=manifest['files'][name],after_sha256=changed_hash,reason='negative control'))
            (root/EVOLUTION).write_text(json.dumps(evolution),encoding='utf-8')
            manifest['files'][name]=changed_hash
            manifest['files'][EVOLUTION]=hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
            manifest['sha256']=digest({k:v for k,v in manifest.items() if k!='sha256'})
            (root/MANIFEST).write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'observer changed algorithm AST'):verify(root)
