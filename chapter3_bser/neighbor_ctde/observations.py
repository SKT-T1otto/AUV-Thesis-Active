"""On-board fusion of received sonar, searched cells and accepted tasks."""
from collections import defaultdict
import copy
import numpy as np


PEER_WIDTH = 15
CONTEXT_DIM = 3 * PEER_WIDTH + 8


class SharedEvidence:
    def __init__(self, view):
        planner = view.unwrapped.map_module
        self.scans = defaultdict(dict)
        # Native updates clip log odds. Replay delayed data in measurement
        # order so arrival order cannot change the result or roll back cell ages.
        self.checkpoints = {}
        self.centers = planner.flat_xyz_centers.detach().cpu().numpy().copy()
        self.last_peer_search = np.full(len(self.centers), -1, dtype=int)
        self.half_life = planner.target_revisit_half_life_steps
        self.received_scans = 0
        self.received_checks = 0

    def before_sensing(self, view, frame, arrivals):
        earliest = frame.step
        for report in arrivals:
            if report.origin == view.agent or report.step >= frame.step:
                raise ValueError('invalid foreign evidence or acquisition clock')
            for ordinal, (when, scan) in enumerate(report.scans):
                if scan.agent_ids != (report.origin,) or when > report.step:
                    raise ValueError('invalid sonar source')
                key = (report.origin, ordinal)
                if key in self.scans[when]:
                    raise ValueError('duplicate sonar reached map integration')
                self.scans[when][key] = scan
                earliest = min(earliest, when)
                self.received_scans += 1
            if report.checked_clear:
                spec = view.unwrapped.agent_specs[report.origin]
                covered = np.linalg.norm(self.centers - report.position, axis=1) <= spec['sensor_range']
                self.last_peer_search[covered] = np.maximum(self.last_peer_search[covered], report.step)
                self.received_checks += 1
        if earliest < frame.step:
            start = max(t for t in self.checkpoints if t <= earliest)
            planner = copy.deepcopy(self.checkpoints[start])
            for t in range(start, frame.step):
                if t > start and t % 8 == 0:
                    self.checkpoints[t] = copy.deepcopy(planner)
                planner.set_mapping_step(t)
                for key in sorted(self.scans[t]):
                    scan = self.scans[t][key]
                    planner.integrate_obstacle_scan(scan.origins, scan.directions, scan.distances,
                                                    scan.hit_mask, current_step=t)
            view.unwrapped.map_module = planner
        if frame.step % 8 == 0:
            self.checkpoints[frame.step] = copy.deepcopy(view.unwrapped.map_module)
        # The native LocalNavigator now consumes these own measurements once.
        for ordinal, (when, scan) in enumerate(frame.scans):
            if when != frame.step:
                raise ValueError('native own sonar acquisition clock mismatch')
            self.scans[when][view.agent, ordinal] = scan

    def context(self, navigator, inbox):
        i, f = navigator.agent, navigator.env.frame
        data = np.zeros(CONTEXT_DIM, np.float32)
        p = np.array(f.position)
        others = [j for j in range(4) if j != i]
        for slot, j in enumerate(others):
            item = inbox.latest.get(j)
            if item is None:
                continue
            start = slot * PEER_WIDTH
            row = data[start:start + PEER_WIDTH]
            if item.step >= f.step:
                raise ValueError('policy would observe an undelivered task')
            row[0] = 1.
            row[1] = (f.step - item.step) / navigator.horizon
            row[2:5] = (np.asarray(item.position) - p) / navigator.bounds
            spec = navigator.env.unwrapped.agent_specs[j]
            row[5:8] = np.asarray(item.velocity) / [spec['v_xy_max'], spec['v_xy_max'], spec['v_z_max']]
            row[8] = float(item.active)
            if item.root is not None:
                row[9:12] = (np.asarray(item.root) - p) / navigator.bounds
            if item.remaining_route:
                row[12:15] = (np.asarray(item.remaining_route[0]) - p) / navigator.bounds
        # Eight octants summarize timestamped, actual peer detector checks.
        # Traversing a cell alone never creates negative target evidence.
        seen = self.last_peer_search >= 0
        if seen.any():
            octants = ((self.centers >= p) @ np.array([1, 2, 4])).astype(int)
            weights = np.exp2(-(f.step - self.last_peer_search[seen]) / self.half_life)
            sums = np.bincount(octants[seen], weights=weights, minlength=8)
            data[-8:] = sums / np.maximum(1, np.bincount(octants, minlength=8))
        return data
