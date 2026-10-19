from dataclasses import replace
import hashlib
import numpy as np
from core.mapping.planning_state import extract_planning_state
from chapter3_bser.safety.public_geometry import filter_planning_state

def start_status(state, agent_id):
    agent = next((a for a in state.agents if a.agent_id == agent_id))
    point = np.asarray(agent.position, dtype=np.float64)
    role = str(agent.role).lower().startswith('exec')
    for endpoint in state.planning_graph.endpoint_connectors:
        if str(endpoint.role).lower().startswith('exec') == role and np.allclose(endpoint.point, point, atol=1e-12, rtol=0):
            if not endpoint.point_valid:
                return 'invalid_start'
            return 'ready' if endpoint.connectors else 'no_start_connector'
    graph = state.planning_graph
    squared = np.sum((np.asarray(graph.cell_centers) - point) ** 2, axis=1)
    nearest = int(np.argmin(squared))
    if squared[nearest] <= 1e-18:
        return 'ready' if graph.valid_mask[nearest] and graph.component_labels[nearest] >= 0 else 'invalid_start'
    return 'stale_start_endpoint'

def topology_key(state):
    """Risk/geometry content, not the monotonically increasing map counter."""
    h = hashlib.sha256()
    h.update(np.asarray(state.occupancy.occupied_mask, dtype=np.bool_).tobytes())
    h.update(np.asarray(state.planning_graph.valid_mask, dtype=np.bool_).tobytes())
    return h.hexdigest()

class PlanningViews:

    def __init__(self, runtime, clearance):
        self.runtime = runtime
        self.clearance = float(clearance)
        self.live_step = None
        self.live_state = None
        self.filtered_key = None
        self.filtered_graph = None
        self.forced_step = None

    def live(self):
        step = self.runtime.step
        if self.live_step != step:
            self.live_state = extract_planning_state(self.runtime.env)
            self.live_step = step
        return self.live_state

    def refresh(self):
        if self.forced_step != self.runtime.step:
            self.runtime.state = self.runtime.provider.snapshot(force=True)
            self.forced_step = self.runtime.step
            self.runtime.safe_counts['forced_planning_refreshes'] += 1
            self.live_step = self.runtime.step
            self.live_state = self.runtime.state
        return self.runtime.state

    def search(self, state):
        if not self.runtime.safe_options['path_safety'] or state.target_found:
            return state
        key = state.planning_graph.graph_sha256
        if key != self.filtered_key:
            self.filtered_graph = filter_planning_state(state, self.clearance).planning_graph
            self.filtered_key = key
        return replace(state, planning_graph=self.filtered_graph)
