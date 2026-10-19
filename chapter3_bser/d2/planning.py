from dataclasses import dataclass
from dataclasses import replace
import numpy as np
from core.mapping.planning_state import GridGeometryView
from chapter3_bser.safety.planning_refresh import PlanningViews

def locked(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

@dataclass(frozen=True)
class SafetyOccupancy:
    occupied_mask: np.ndarray

@dataclass(frozen=True)
class SafetyState:
    step: int
    grid: GridGeometryView
    occupancy: SafetyOccupancy
    agents: tuple

class D2PlanningViews(PlanningViews):
    """Read current public geometry, never refresh the decision graph implicitly.

    Content keys include exact occupied cells, grid coordinates/dimensions and
    current agent fields. They deliberately do not rely on the map counter.
    State and geometry are local to this runtime and survive snapshot/restore.
    There are no references to hidden targets or ground-truth obstacle objects.
    """

    def __init__(self, runtime, clearance):
        super().__init__(runtime, clearance)
        self._safety_key = None
        self._grid_key = None
        self._grid = None

    def live(self):
        env = self.runtime.env
        (task, public) = (env.get_task_state(), env.get_agent_state())
        planner = env.unwrapped.map_module
        centers = np.asarray(planner.flat_xyz_centers.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 3)
        shape = tuple((int(x) for x in planner.grid_size))
        spacing = (float(planner.cell_dx), float(planner.cell_dy), float(planner.cell_dz))
        grid_key = (shape, spacing, centers.tobytes())
        if grid_key != self._grid_key:
            self._grid = GridGeometryView(shape, tuple((float(centers[0, a] - 0.5 * spacing[a]) for a in range(3))), spacing, locked(centers, np.float64))
            self._grid_key = grid_key
        occupied = planner.known_occupied_mask if hasattr(planner, 'known_occupied_mask') else ~planner.valid_mask
        occupied = np.asarray(occupied.detach().cpu().numpy(), dtype=np.bool_).reshape(-1)
        agents = tuple((replace(old, position=tuple((float(x) for x in public.positions[old.agent_id])), velocity=tuple((float(x) for x in public.velocities[old.agent_id])), current_navigation_target=tuple((float(x) for x in public.navigation_targets[old.agent_id]))) for old in self.runtime.state.agents))
        key = (int(task.step), grid_key, occupied.tobytes(), agents)
        if key != self._safety_key:
            self.live_state = SafetyState(int(task.step), self._grid, SafetyOccupancy(locked(occupied, np.bool_)), agents)
            (self.live_step, self._safety_key) = (int(task.step), key)
        return self.live_state

    def refresh(self):
        state = super().refresh()
        self._safety_key = None
        return state
