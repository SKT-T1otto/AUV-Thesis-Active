from copy import deepcopy
import json
from project_paths import ROOT

def build_ch3_config(base_candidate, scenario_profile):
    if base_candidate != 'ch3_v3_full_reference' or scenario_profile != 'M20_MOVING_UNKNOWN_MULTI':
        raise ValueError('active extraction supports only the audited environment/profile')
    return deepcopy(json.loads((ROOT / 'configs/environment.json').read_text(encoding='utf-8')))
