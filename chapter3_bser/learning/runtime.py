from collections import Counter
from chapter3_bser.learning.base_runtime import BaselineMissionRuntime
from chapter3_bser.learning.gated_runtime import RepairedMissionRuntime as ParentRuntime
from chapter3_bser.learning.config import VERSION
from chapter3_bser.learning.config import validate

class RepairedMissionRuntime(ParentRuntime):

    def __init__(self, config, *args, **kwargs):
        self.repair_config = validate(config)
        self.repair_counts = Counter()
        self.raw_components = {}
        self.phase_returns = Counter()
        BaselineMissionRuntime.__init__(self, self.repair_config, *args, **kwargs)

    def _gates(self):
        gates = super()._gates()
        options = self.config.get('map_experiment', {}).get('executor_safety')
        if (options is not None and options.get('schema') == 'ch3.executor_recovery.v2'
                and options['enabled'] and options['executor_search_residual'] == 'disabled'
                and not self.env.get_task_state().target_found):
            gates[3] = False
        return gates

    def summary(self):
        row = super().summary()
        row['repair_version'] = VERSION
        return row
