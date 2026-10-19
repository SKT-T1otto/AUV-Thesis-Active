from dataclasses import dataclass
from enum import Enum
from chapter3_bser.online.controller import OnlineBSERController
CONTROLLER_FACTORY_VERSION = 'prrac.controller_factory.v1'
CHECKPOINT_RUNTIME_REVISION = 'dynamic_public_intercept_v2_1'

class ExecutionVariant(str, Enum):
    B0_LEGACY_V2_1 = 'B0_LEGACY_V2_1'

@dataclass(frozen=True)
class PRRACRuntimeContract:
    checkpoint_runtime_revision: str
    execution_variant: ExecutionVariant
    runtime_integration_mode: str
    controller_factory_version: str = CONTROLLER_FACTORY_VERSION

def build_prrac_online_controller(phase1b_config, experiment_config, **kwargs):
    if kwargs or experiment_config.get('execution_variant', 'B0_LEGACY_V2_1') != 'B0_LEGACY_V2_1' or experiment_config.get('runtime_integration_mode', 'legacy') != 'legacy':
        raise ValueError('active extraction contains only the audited D2 legacy controller protocol')
    if experiment_config.get('search_value_guidance', {}).get('enabled') or experiment_config.get('early_discovery', {}).get('enabled'):
        raise ValueError('unused learned scoring/BEDS variants were not extracted')
    controller = OnlineBSERController(dict(phase1b_config))
    controller.prrac_runtime_contract = PRRACRuntimeContract(CHECKPOINT_RUNTIME_REVISION, ExecutionVariant.B0_LEGACY_V2_1, 'legacy')
    return controller
