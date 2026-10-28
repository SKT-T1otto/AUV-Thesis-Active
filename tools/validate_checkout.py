"""Verify package/Git bytes, then run the bounded engineering checks manually."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from project_paths import ROOT


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_package(root):
    root = Path(root).resolve()
    files = json.loads((root / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    if not isinstance(files, dict) or not files:
        raise ValueError('empty or invalid package inventory')
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name == 'PACKAGE_SHA256.json':
            raise ValueError('unsafe package path: ' + name)
        if not path.is_file() or sha(path.read_bytes()) != expected:
            raise ValueError('package file missing or changed: ' + name)
    return dict(passed=True, files=len(files), package_sha256=sha((root / 'PACKAGE_SHA256.json').read_bytes()))


def git(root, *args, data=None):
    result = subprocess.run(['git', '-c', 'core.fsmonitor=false', '-C', str(root), *args],
                            input=data, capture_output=True, timeout=120)
    if result.returncode:
        raise ValueError('git failed: ' + ' '.join(args) + '\n' + result.stderr.decode('utf-8', errors='replace'))
    return result.stdout


def verify_index(root):
    """Check actual staged blobs, not only working files (no commit is made)."""
    root = Path(root).resolve()
    verify_package(root)
    package = json.loads((root / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    package['PACKAGE_SHA256.json'] = sha((root / 'PACKAGE_SHA256.json').read_bytes())
    entries = []
    for record in git(root, 'ls-files', '--stage', '-z').split(b'\0'):
        if not record:
            continue
        meta, name = record.split(b'\t', 1)
        mode, oid, stage = meta.split()
        if mode not in (b'100644', b'100755') or stage != b'0':
            raise ValueError('unsupported or unmerged Git entry')
        entries.append((name.decode('utf-8'), oid))
    if len(entries) != len(package) or {name for name, _ in entries} != set(package):
        raise ValueError('Git index must contain exactly the package files; missing files or extra artifacts staged')
    blobs = git(root, 'cat-file', '--batch', data=b''.join(oid + b'\n' for _, oid in entries))
    offset = 0
    for name, oid in entries:
        end = blobs.index(b'\n', offset)
        header_oid, kind, size = blobs[offset:end].split()
        start = end + 1
        end = start + int(size)
        if header_oid != oid or kind != b'blob' or blobs[end:end+1] != b'\n':
            raise ValueError('invalid Git blob response')
        if sha(blobs[start:end]) != package[name]:
            raise ValueError('Git staged bytes differ from package: ' + name)
        offset = end + 1
    if offset != len(blobs):
        raise ValueError('unexpected trailing Git blob response')
    return dict(passed=True, files=len(entries), staged_bytes_exact=True)


def verify_clean_commit(root, expected=None):
    if Path(git(root, 'rev-parse', '--show-toplevel').decode('utf-8').strip()).resolve() != Path(root).resolve():
        raise ValueError('must run from the standalone repository, not a parent Git checkout')
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    if expected and (len(expected) != 40 or expected.lower() != head):
        raise ValueError('checkout does not match the specified full commit SHA')
    if git(root, 'status', '--porcelain=v1', '--untracked-files=all').strip():
        raise ValueError('checkout has modified, staged, or untracked files')
    return dict(head=head, clean=True, index=verify_index(root))


def run(output, *, require_linux=False, expected_commit=None,checks_timeout=3600,safe_rl_prefix=None):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'work'):
        raise ValueError('validation output must be under this checkout work/')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='auv.active.checkout.validation.v1', passed=False,
                  python=sys.version, executable=sys.executable, platform=platform.platform(),
                  command=sys.argv, performance_passed=False, formal_training_started=False,
                  linux_execution=platform.system() == 'Linux')
    try:
        if require_linux and platform.system() != 'Linux':
            raise ValueError('--require-linux must run on an actual Linux host')
        result['git'] = verify_clean_commit(ROOT, expected_commit)
        result['package_before'] = verify_package(ROOT)
        from chapter3_bser.learning.provenance import verify_reviewed_changes, research_readiness
        source = verify_reviewed_changes()
        research_readiness(source)
        result['source'] = source
        result['versions'] = {name: importlib.metadata.version(name) for name in ('numpy', 'torch')}
        reach_avoid_workflow=source.get('profile') in ('active_reach_avoid_v1','active_reach_avoid_geometry_v2','active_reach_avoid_monitor_v3')
        safety_workflow=source.get('profile') in ('active_safe_rl_transport_v1','active_safe_rl_recovery_v2','active_safe_rl_v1','active_safe_rl_validation_v1')
        diagnostics_workflow=source.get('profile')=='active_executor_diagnostics_v1'
        recovery_workflow=source.get('profile')=='active_executor_recovery_v2'
        standby_workflow=source.get('profile')=='active_executor_standby_v1'
        map_workflow=reach_avoid_workflow or safety_workflow or diagnostics_workflow or recovery_workflow or standby_workflow or source.get('profile') in ('active_map_experiment_v1','active_map_convergence_v1','active_map_persistence_v1','active_map_cache_v1','active_map_bounded_v3','active_map_route_health_v1','active_map_diagnostics_v1')
        command = [sys.executable, '-m', 'tools.run_checks', '--output', str(output / 'checks'), '--execute',
                   '--reach-avoid-workflow' if reach_avoid_workflow else '--safe-rl-workflow' if safety_workflow else '--executor-diagnostics-workflow' if diagnostics_workflow else '--recovery-workflow' if recovery_workflow else '--standby-workflow' if standby_workflow else '--map-workflow' if map_workflow else '--rollout']
        if safe_rl_prefix:
            if not safety_workflow:raise ValueError('prefix continuation requires safety workflow')
            from tools.safe_rl_resume import load_prefix
            load_prefix(safe_rl_prefix)
            command+=['--safe-rl-prefix',str(Path(safe_rl_prefix).resolve())]
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
                   PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        from tools.check_process import run_checks_process
        print(f'[validation] starting bounded checks; wall timeout={checks_timeout}s',flush=True)
        result['child']=run_checks_process(command,cwd=ROOT,env=env,output=output,timeout=checks_timeout)
        if result['child']['timed_out']:raise TimeoutError('bounded checks wall timeout; partial stdout/stderr retained')
        if result['child']['exit_code']:
            raise RuntimeError('bounded checks failed; see checks.stdout.log and checks.stderr.log')
        checks = json.loads((output / 'checks/checks.json').read_text(encoding='utf-8'))
        expected = json.loads((ROOT / 'configs/rebuild_check_contract.json').read_text(encoding='utf-8'))['expected_tests']
        regression = checks['regression']
        if (checks.get('passed') is not True or regression['tests_run'] != expected
                or regression['failures'] or regression['errors'] or regression['skipped']
                or checks.get('workflow', {}).get('passed') is not True):
            raise ValueError('incomplete bounded verification evidence')
        result['checks'] = dict(passed=True, tests_run=expected, additional_physical_steps=checks['workflow'].get('additional_physical_steps',checks['workflow'].get('physical_steps')),
            environment_initializations=checks['workflow'].get('environment_initializations'),
            additional_environment_initializations=checks['workflow'].get('additional_environment_initializations'),
            max_complete_updates_per_model=checks['workflow']['max_complete_updates_per_model'])
        result['map_workflow_verified']=map_workflow and checks['workflow'].get('checkpoint_file_restore_executed') is False
        if verify_reviewed_changes() != source:
            raise ValueError('source changed during validation')
        research_readiness(source)
        result['package_after'] = verify_package(ROOT)
        result['git_after'] = verify_clean_commit(ROOT, expected_commit)
        result['passed'] = True
    except Exception:
        result['traceback'] = traceback.format_exc()
    (output / 'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--index-only', action='store_true')
    parser.add_argument('--require-linux', action='store_true')
    parser.add_argument('--expected-commit')
    parser.add_argument('--checks-timeout',type=int,default=3600)
    parser.add_argument('--safe-rl-prefix')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(executed=False, **vars(args))))
        return
    if args.index_only:
        print(json.dumps(verify_index(ROOT), indent=2))
        return
    if not args.output:
        parser.error('--output is required for full validation')
    result = run(args.output, require_linux=args.require_linux, expected_commit=args.expected_commit,
        checks_timeout=args.checks_timeout,safe_rl_prefix=args.safe_rl_prefix)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
