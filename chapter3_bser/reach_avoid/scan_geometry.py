"""Public scan subcells refine coarse planning occupancy for local control.

Coarse occupied cells remain planning risk, not physical collision events.
Only ray-traversed 0.25m subcells without a recorded hit can override their
local motion veto. Unobserved parts remain blocked; hit padding and the
finite motion/braking model remain active. Finite rays and voxelization are
an approximation, never a certificate of continuous-world free space.
"""
import numpy as np
from itertools import product
from chapter3_bser.safety.public_geometry import PublicGeometry
from chapter3_bser.safe_rl.evidence import EvidenceGeometry
from chapter3_bser.safe_rl.buffer_recovery import ExitCorridor, velocity_command
from chapter3_bser.safe_rl.dynamics import integrate, brake

REVISION = 'public_scan_subcells_v1'
RESOLUTION = .25


def segment_pieces(start, end, resolution=RESOLUTION):
    """Split at every subcell face, including zero-length endpoint tests."""
    p, q = np.asarray(start, float), np.asarray(end, float)
    d = q - p
    times = [0., 1.]
    for axis in range(3):
        if abs(d[axis]) < 1e-15:
            continue
        lo, hi = sorted((p[axis], q[axis]))
        planes = np.arange(np.floor(lo / resolution) + 1, np.ceil(hi / resolution)) * resolution
        times.extend(((planes - p[axis]) / d[axis]).tolist())
    t = np.unique(np.clip(times, 0., 1.))
    starts, ends = p + t[:-1, None] * d, p + t[1:, None] * d
    starts = np.concatenate((p[None], starts, q[None]))
    ends = np.concatenate((p[None], ends, q[None]))
    keys = np.floor(((starts + ends) * .5) / resolution).astype(np.int64)
    return starts, ends, keys


def ray_cells(start, end):
    """Closed voxel supercover: a ray on a face visits BOTH adjacent cells."""
    starts, ends, keys = segment_pieces(start, end)
    scaled = (starts+ends)*.5/RESOLUTION
    face = np.isclose(scaled, np.rint(scaled), rtol=0., atol=1e-10)
    base = np.where(face, np.rint(scaled), keys).astype(int)
    cells = set(map(tuple, base))
    for offset in product(*[(-1,0) if present else (0,) for present in face.any(axis=0)]):
        if not any(offset):
            continue
        offset = np.asarray(offset)
        rows = np.all((offset == 0) | face, axis=1)
        cells.update(map(tuple, base[rows]+offset))
    return cells


class RaySubcells:
    """Episode-local static-obstacle evidence; a hit always overrides free."""
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, float)
        self.shape = tuple(np.ceil(self.bounds/RESOLUTION).astype(int))
        self.free = set()
        self.blocked = set()
        self.traversed = set()
        self.cursor = 0

    def _inside(self, key):
        return all(0 <= k < n for k,n in zip(key,self.shape))

    def consume(self, events, step):
        free, blocked = set(), set()
        for event in events:
            if event.sequence <= self.cursor or event.physical_step > step:
                raise ValueError('subcell evidence clock regressed or is in the future')
            self.cursor = event.sequence
            if event.kind != 'scan':
                continue
            scan = event.payload
            directions = np.asarray(scan.directions, float)
            for origin, distances, hits in zip(scan.origins, scan.distances, scan.hit_mask):
                origin = np.asarray(origin, float)
                for direction, distance, hit in zip(directions, distances, hits):
                    # Leave half a subcell before the measured surface. The
                    # terminal subcell is separately blocked even on a tie.
                    length = max(0., float(distance) - (.5 * RESOLUTION if hit else 0.))
                    keys = ray_cells(origin, origin + direction * length)
                    free.update(k for k in keys if self._inside(k))
                    if hit:
                        key = tuple(np.floor((origin + direction * float(distance)) / RESOLUTION).astype(int))
                        if self._inside(key):
                            blocked.add(key)
        self.blocked.update(blocked)
        self.traversed.update(free)
        self.free.update(free)
        self.free.difference_update(self.blocked)

    def contains(self, points):
        keys = np.floor(np.asarray(points, float).reshape(-1, 3) / RESOLUTION).astype(int)
        return np.array([tuple(k) in self.free for k in keys], bool)


class ScanRefinedGeometry(EvidenceGeometry):
    def __init__(self, state, evidence, options, subcells):
        super().__init__(state, evidence, options)
        self.subcells = subcells
        self._coarse = {}

    def _coarse_points(self, point):
        closest = np.clip(np.asarray(point, float), self.raw.occupied_lower, self.raw.occupied_upper)
        # This removes a coarse distance/proxy veto only where public rays
        # support finer free space. Segment checks still inspect EVERY piece.
        return closest[~self.subcells.contains(closest)]

    def directional_points(self, point):
        return np.concatenate((self.points, self._coarse_points(point)), axis=0)

    def point_distances(self, point):
        p = np.asarray(point, float)
        hit = float(np.linalg.norm(self.points - p, axis=1).min()) if len(self.points) else float('inf')
        closest = self._coarse_points(p)
        voxel = float(np.linalg.norm(closest - p, axis=1).min()) if len(closest) else float('inf')
        return hit, voxel

    def observed_path(self, start, end, extra=0., *, egress_origin=None):
        """Coarse occupied/padded portions require actual free-ray subcells."""
        p, q = np.asarray(start, float), np.asarray(end, float)
        if not np.isfinite([p, q]).all() or np.any(np.minimum(p, q) < self.raw.lower) or np.any(np.maximum(p, q) > self.raw.upper):
            return False
        starts, ends, keys = segment_pieces(p, q)
        initial = None if egress_origin is None else tuple(np.floor(np.asarray(egress_origin)/RESOLUTION).astype(int))
        # Only the separate, outward-supporting-plane egress probe may leave
        # its initially mixed hit/free subcell. Ordinary motion never gets
        # this exception. The rest of the corridor still needs free evidence.
        initial_exit = np.array([tuple(k)==initial and tuple(k) in self.subcells.traversed for k in keys])
        if any(tuple(k) in self.subcells.blocked and not initial_exit[j] for j,k in enumerate(keys)):
            return False
        unknown = np.array([tuple(k) not in self.subcells.free for k in keys]) & ~initial_exit
        if not unknown.any():
            return True
        margin = self.padding + float(extra)
        if margin not in self._coarse:
            self._coarse[margin] = PublicGeometry(self.state, margin)
        return bool(self._coarse[margin]._segments_free(starts[unknown], ends[unknown]).all())

    def segment(self, start, end, extra=0.):
        if not self.observed_path(start, end, extra):
            return False
        p, q = np.asarray(start, float), np.asarray(end, float)
        if len(self.points):
            d = q - p
            t = np.clip((self.points - p) @ d / max(float(d @ d), 1e-30), 0., 1.)
            gap = np.linalg.norm(self.points - (p + t[:, None] * d), axis=1)
            if np.any(gap <= self.padding + extra):
                return False
        return True

    def local(self, point):
        unknown, coarse_free = super().local(point)
        return unknown, bool(coarse_free or self.subcells.contains([point])[0])

    def diagnostics(self, point):
        return dict(revision=REVISION, subcell_resolution=RESOLUTION,
            pose_ray_free=bool(self.subcells.contains([point])[0]),
            coarse_pose_blocked=not self.raw.point_free(point),
            free_subcells=len(self.subcells.free), hit_subcells=len(self.subcells.blocked),
            collision_semantics='environment_segment_closed_true_aabb',
            continuous_free_space_certified=False)


def refined_buffer_exit(p, v, spec, geometry, contract, options, flow):
    """Retain parent outward check, also enforce the finer observed corridor."""
    if geometry.distance(p) > geometry.padding + .10:
        return None
    corridor = ExitCorridor(geometry, p, extra_margin=.10)
    if not corridor.valid or not len(corridor.normals):
        return None
    dt = contract['dt']
    origin = np.asarray(p, float)
    bound = options['flow_component_bound']
    flows = [np.asarray(flow, float), *[np.array(x) for x in product((-bound, bound), repeat=2)]]
    directions = [corridor.normals.sum(0), *corridor.normals, *np.eye(3), *-np.eye(3)]
    for direction in directions:
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction = direction / norm
        if np.any(corridor.normals @ direction < 1e-8):
            continue
        desired = .35 * direction
        command = velocity_command(v, desired, spec, dt, flow)
        passed, probes = True, []
        for disturbance in flows:
            x, vel = origin.copy(), np.asarray(v, float).copy()
            exited = stopped = False
            for k in range(options['horizon']):
                if geometry.distance(x) > geometry.padding + .15:
                    exited = True
                u = command if k == 0 else brake(vel, spec, dt, disturbance) if exited else velocity_command(vel, desired, spec, dt, disturbance)
                nxt, nv = integrate(x, vel, u, spec, dt, disturbance)
                if not corridor.segment(x, nxt) or not geometry.observed_path(x, nxt, egress_origin=origin):
                    passed = False
                    break
                if k == 0 and (np.min(corridor.normals @ (nxt - origin)) <= 1e-7 or np.linalg.norm(nv[:2]) > .5 or abs(nv[2]) > .4):
                    passed = False
                    break
                x, vel = nxt, nv
                if exited and np.linalg.norm(vel) < .04:
                    stopped = True
                    break
            if not passed or not stopped:
                passed = False
                break
            probes.append(dict(flow=disturbance.tolist(), steps=k+1, stop=x.tolist()))
        if passed:
            return command, dict(reason='outward_margin_exit_and_stop', probes=probes,
                flow_probe_model='constant bounded XY flow, feedback after first step',
                origin_in_occupied_voxel=bool(np.any(corridor.enclosing)),
                public_subcell_corridor_checked=True, terminal_invariant_certified=False,
                ground_truth_geometry_used=False, isotropic_zero_cap_exemption=True)
    return None
