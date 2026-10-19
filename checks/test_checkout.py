"""Transport validation negative controls; no simulator or user artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools.validate_checkout import verify_package, verify_index, verify_clean_commit, run


class Checkout(unittest.TestCase):
    def make_fixture(self, root, attributes=b'* -text\n'):
        (root / '.gitattributes').write_bytes(attributes)
        (root / 'frozen.json').write_bytes(b'{"immutable": true}\r\n')
        inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
        (root / 'PACKAGE_SHA256.json').write_text(json.dumps(inventory), encoding='utf-8')
        subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
        subprocess.run(['git', '-c', 'core.autocrlf=true', '-C', str(root), 'add', '.'], check=True, capture_output=True)

    def test_package_missing_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_fixture(root)
            verify_package(root)
            (root / 'frozen.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                verify_package(root)

    def test_package_changed_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_fixture(root)
            (root / 'frozen.json').write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                verify_package(root)

    def test_index_preserves_crlf_with_autocrlf_enabled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_fixture(root)
            self.assertTrue(verify_index(root)['staged_bytes_exact'])

    def test_normalizing_attributes_rejected_before_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_fixture(root, b'* text eol=lf\n')
            with self.assertRaisesRegex(ValueError, 'staged bytes differ'):
                verify_index(root)

    def test_extra_staged_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_fixture(root)
            (root / 'accidental.pt').write_bytes(b'not a user model')
            subprocess.run(['git', '-C', str(root), 'add', 'accidental.pt'], check=True, capture_output=True)
            with self.assertRaisesRegex(ValueError, 'exactly the package'):
                verify_index(root)

    def test_wrong_commit_rejected_before_tests(self):
        root = Path('/fixture').resolve()
        with patch('tools.validate_checkout.git', side_effect=[str(root).encode('utf-8') + b'\n', b'1' * 40 + b'\n']):
            with self.assertRaisesRegex(ValueError, 'specified full commit'):
                verify_clean_commit(root, '2' * 40)

    def test_linux_flag_cannot_claim_windows_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('tools.validate_checkout.ROOT', root), patch('tools.validate_checkout.platform.system', return_value='Windows'):
                result = run(root / 'work/result', require_linux=True)
            self.assertFalse(result['passed'])
            self.assertFalse(result['linux_execution'])
            self.assertIn('actual Linux host', result['traceback'])


if __name__ == '__main__':
    unittest.main()
