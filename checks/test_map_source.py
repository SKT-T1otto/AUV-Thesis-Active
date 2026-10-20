"""Independent evolution negative controls; no runtime or archived-code imports."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from project_paths import ROOT
from tools.map_convergence_source import verify, digest, BASELINE, EVOLUTION, MANIFEST


class MapSource(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        manifest = json.loads((ROOT/MANIFEST).read_text(encoding='utf-8'))
        baseline = json.loads((ROOT/BASELINE/'baseline.json').read_text(encoding='utf-8'))
        names = set(manifest['files']) | {MANIFEST, 'provenance/active_manifest.json'}
        names.update(name for name in baseline['files'] if name.startswith('verification/'))
        for name in names:
            target = root/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        return root, manifest

    def test_missing_new_module_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            (root/'core/env/public_events.py').unlink()
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'): verify(root)

    def test_parent_archive_and_old_evidence_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            evidence = root/'verification/rebuild/receipt.json'
            evidence.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'parent evidence changed'): verify(root)

    def test_resigned_manifest_cannot_hide_a_physics_change(self):
        with tempfile.TemporaryDirectory() as directory:
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
