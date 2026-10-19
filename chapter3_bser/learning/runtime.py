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

    def summary(self):
        row = super().summary()
        row['repair_version'] = VERSION
        return row
