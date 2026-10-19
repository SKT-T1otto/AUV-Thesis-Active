from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.hysteresis.policy import ReplanningPolicy

class SearchFailurePolicy(ReplanningPolicy):
    RECOVERY_EVENTS = {BSEREvent.WAYPOINT_STALE, BSEREvent.OBSTACLE_DISCOVERED}

    def __init__(self, config):
        super().__init__(config)
        self.search_enabled = True
        self.recovery_ready = False
        self.last_failed_step = None
        self.last_failed_topology = None
        self.retry_steps = int(config['hysteresis']['waypoint_stale_cooldown_steps'])
        self.retry_remaining = 0

    def ready(self, state, topology, *, full_refresh=False):
        return bool(self.last_failed_step is None or full_refresh or topology != self.last_failed_topology or (state.step - self.last_failed_step >= self.retry_steps))

    def failed(self, step, topology):
        self.last_failed_step = int(step)
        self.last_failed_topology = topology

    def succeeded(self):
        self.last_failed_step = None
        self.last_failed_topology = None

    def event_cooldown_remaining(self, event, step):
        if self.search_enabled and self.recovery_ready and (event in self.RECOVERY_EVENTS):
            return 0
        if self.search_enabled and event in self.RECOVERY_EVENTS and self.retry_remaining:
            return self.retry_remaining
        return super().event_cooldown_remaining(event, step)

    def mark_attempt(self, step, event):
        if not self.search_enabled or event not in self.RECOVERY_EVENTS:
            super().mark_attempt(step, event)
