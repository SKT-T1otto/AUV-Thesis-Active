"""Independent filesystem verifier; imports no production implementation."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    if (root/'provenance/safe_rl_recovery_manifest.json').is_file():
        from tools.safe_rl_recovery_source import verify as verify_recovery
        return verify_recovery(root)
    if (root/'provenance/safe_rl_validation_manifest.json').is_file():
        from tools.safe_rl_validation_source import verify as verify_safety_validation
        return verify_safety_validation(root)
    if (root/'provenance/safe_rl_manifest.json').is_file():
        from tools.safe_rl_source import verify as verify_safety
        return verify_safety(root)
    if (root/'provenance/executor_diagnostics_manifest.json').is_file():
        from tools.executor_diagnostics_source import verify as verify_executor_diagnostics
        return verify_executor_diagnostics(root)
    if (root/'provenance/executor_recovery_manifest.json').is_file():
        from tools.executor_recovery_source import verify as verify_recovery
        return verify_recovery(root)
    if (root/'provenance/executor_standby_manifest.json').is_file():
        from tools.executor_standby_source import verify as verify_standby
        return verify_standby(root)
    if (root/'provenance/map_diagnostics_manifest.json').is_file():
        from tools.map_diagnostic_source import verify as verify_diagnostics
        return verify_diagnostics(root)
    if (root/'provenance/map_route_health_manifest.json').is_file():
        from tools.map_route_health_source import verify as verify_health
        return verify_health(root)
    if (root/'provenance/map_bounded_manifest.json').is_file():
        from tools.map_bounded_source import verify as verify_bounded
        return verify_bounded(root)
    if (root/'provenance/map_cache_manifest.json').is_file():
        from tools.map_cache_source import verify as verify_cache
        return verify_cache(root)
    if (root/'provenance/map_persistence_manifest.json').is_file():
        from tools.map_persistence_source import verify as verify_map
        return verify_map(root)
    if (root/'provenance/map_convergence_manifest.json').is_file():
        from tools.map_convergence_source import verify as verify_map
        return verify_map(root)
    if (root/'provenance/map_experiment_manifest.json').is_file():
        from tools.map_experiment_source import verify as verify_map
        return verify_map(root)
    if (root/'provenance/map_core_manifest.json').is_file():
        from tools.map_core_source import verify as verify_map
        return verify_map(root)
    if (root/'provenance/map_stage1_manifest.json').is_file():
        from tools.map_interface_source import verify as verify_map
        return verify_map(root)
    data = json.loads((root/'provenance/active_manifest.json').read_text(encoding='utf-8'))
    if data.get('schema') != 'auv.active_source.rebuild.v1' or data.get('sha256') != digest({k:v for k,v in data.items() if k!='sha256'}):
        raise ValueError('invalid rebuild manifest identity')
    actual = {}
    for folder in ('core','chapter3_bser','configs','checks','tools','provenance/parent','provenance/history'):
        for file in (root/folder).rglob('*'):
            if file.is_file() and file.suffix in ('.py','.json'):
                actual[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    for name in ('cli.py','project_paths.py','requirements.txt','AGENTS.md','.gitattributes','.gitignore','provenance/rebuild_evolution.json'):
        actual[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != data['files']:
        raise ValueError('independent source inventory mismatch: '+repr(sorted(k for k in actual.keys()|data['files'].keys() if actual.get(k)!=data['files'].get(k))))
    old = json.loads((root/'provenance/history/extraction_v1/active_manifest.json').read_text(encoding='utf-8'))
    if old['sha256'] != data['previous_manifest_sha256']:
        raise ValueError('wrong extraction parent')
    if any(actual.get(k)!=v for k,v in old['files'].items() if k.startswith('core/')):
        raise ValueError('rebuild changed shared core behavior')
    evolution = json.loads((root/'provenance/rebuild_evolution.json').read_text(encoding='utf-8'))
    rows = evolution['changes']; changes = {r['path']:r for r in rows}
    required = {k for k in old['files'].keys()|actual.keys() if old['files'].get(k)!=actual.get(k)}-{'provenance/rebuild_evolution.json'}
    if len(changes)!=len(rows) or set(changes)!=required:
        raise ValueError('evolution coverage mismatch')
    for name,row in changes.items():
        if row['before_sha256']!=old['files'].get(name) or row['after_sha256']!=actual.get(name) or not row['reason']:
            raise ValueError('evolution hash/reason mismatch: '+name)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],core_unchanged=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    print(json.dumps(verify(parser.parse_args().root),indent=2))


if __name__=='__main__':main()
