import json
from typing import Dict
from typing import Any
from project_paths import ROOT
EXECUTION_RUNTIME_DEFAULTS = {'dynamic_public_target_enabled': False, 'public_target_update_distance': 0.75, 'public_target_update_min_steps': 20, 'defer_stale_endpoint_invalid': False, 'refresh_on_executor_handoff': False, 'refresh_on_public_target_shift': False}

def execution_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = dict(EXECUTION_RUNTIME_DEFAULTS)
    runtime.update(dict(config.get('execution_runtime', {})))
    if float(runtime['public_target_update_distance']) < 0.0:
        raise ValueError('public_target_update_distance must be non-negative')
    if int(runtime['public_target_update_min_steps']) < 0:
        raise ValueError('public_target_update_min_steps must be non-negative')
    return runtime

def load_phase1b2_config():
    return json.loads((ROOT / 'configs/online.json').read_text(encoding='utf-8'))

def load_phase1b_config(path=None):
    if path is not None:
        raise ValueError('only the extracted active online configuration is supported')
    return load_phase1b2_config()
