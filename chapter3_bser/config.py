import json
from project_paths import ROOT

def load_bser_phase1a1_config(path=None):
    if path is not None:
        raise ValueError('only the extracted active allocation configuration is supported')
    value = json.loads((ROOT / 'configs/allocation.json').read_text(encoding='utf-8'))
    if value.get('schema') != 'bser.phase1a1.config.v1' or value.get('formal_training') is not False or value['e1_v2']['snapshot_steps'] != [0, 10, 25, 50]:
        raise ValueError('invalid extracted allocation contract')
    return value
