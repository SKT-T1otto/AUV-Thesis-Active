"""Validate and deduplicate public transactions without receiving an environment."""
from collections import OrderedDict
import numpy as np
from core.env.public_events import PublicEvent, ScanObservation, StepObservation, payload_digest
from .types import PublicStepEvidence


def _finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('invalid public ' + name)
    return array


class PublicObservationAdapter:
    def __init__(self, *, capacity=1024):
        if type(capacity) is not int or capacity < 2:
            raise ValueError('invalid observation history capacity')
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.episode_id = None
        self.cursor = 0
        self.physical_step = 0
        self.completed_step = 0
        self.last_scan_step = -1
        self.found = False
        self.terminated = False
        self._seen = OrderedDict()

    def ingest(self, events):
        # Validate on a local copy: an invalid batch must not partly consume evidence.
        episode, cursor, step, completed = self.episode_id, self.cursor, self.physical_step, self.completed_step
        last_scan, found, terminated = self.last_scan_step, self.found, self.terminated
        seen = self._seen.copy()
        groups = OrderedDict()
        for event in events:
            if not isinstance(event, PublicEvent):
                raise ValueError('only typed public events are accepted')
            if (not isinstance(event.payload, (ScanObservation, StepObservation))
                    or type(event.physical_step) is not int or event.physical_step < 0
                    or type(event.sequence) is not int or event.sequence < 1
                    or not isinstance(event.episode_id, str) or not event.episode_id):
                raise ValueError('invalid public event envelope')
            if payload_digest(event.payload) != event.payload_sha256:
                raise ValueError('public event payload hash mismatch')
            if event.event_id in seen:
                if seen[event.event_id] != event.payload_sha256:
                    raise ValueError('conflicting duplicate public event')
                continue
            if terminated:
                raise ValueError('new evidence after terminal transition')
            if episode is None:
                if event.sequence != 1 or event.physical_step != 0 or event.kind != 'scan':
                    raise ValueError('public history must begin with the reset scan')
                episode = event.episode_id
            if event.episode_id != episode:
                raise ValueError('mixed episodes; explicitly reset the observation adapter')
            if event.sequence != cursor + 1 or event.physical_step < step:
                raise ValueError('public event sequence gap or reversed physical time')
            if event.physical_step > completed + 1:
                raise ValueError('missing physical-step completion')
            key = event.physical_step
            group = groups.setdefault(key, [[], None])
            if event.kind == 'scan' and isinstance(event.payload, ScanObservation):
                p = event.payload
                if key != 0 and key <= completed:
                    raise ValueError('scan arrived after step completion')
                if event.source_id != 'shared_neighbor26' or p.agent_ids != (0, 1, 2, 3):
                    raise ValueError('invalid public scan source')
                _finite(p.origins, (4, 3), 'scan origins')
                directions = _finite(p.directions, (26, 3), 'scan directions')
                distances = _finite(p.distances, (4, 26), 'scan distances')
                if np.any(distances < 0) or not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-7):
                    raise ValueError('invalid public scan range/direction')
                hits = np.asarray(p.hit_mask)
                if hits.shape != (4, 26) or hits.dtype != np.bool_:
                    raise ValueError('invalid public hit mask')
                group[0].append(event)
                last_scan = key
            elif event.kind == 'step' and isinstance(event.payload, StepObservation):
                p = event.payload
                if event.source_id != 'team_search_detector' or key != completed + 1:
                    raise ValueError('invalid public transition order')
                for name in ('agent_start', 'agent_end', 'velocity_start', 'velocity_end'):
                    _finite(getattr(p, name), (4, 3), name)
                flags = ('found_before', 'found_after', 'target_transition_completed',
                         'discovery_check_completed', 'episode_terminated')
                if any(type(getattr(p, field)) is not bool for field in flags):
                    raise ValueError('public event flags must be boolean')
                if p.found_before and not p.found_after:
                    raise ValueError('discovery cannot be undone')
                if p.found_before != found:
                    raise ValueError('public discovery history is inconsistent')
                if p.discovery_check_completed and (not p.target_transition_completed or p.found_before):
                    raise ValueError('inconsistent discovery-check validity')
                if p.terminal_reason not in ('running', 'success', 'obstacle_collision', 'timeout'):
                    raise ValueError('invalid public terminal reason')
                if p.episode_terminated != (p.terminal_reason != 'running'):
                    raise ValueError('public termination metadata disagree')
                if p.terminal_reason == 'obstacle_collision' and (p.discovery_check_completed or p.target_transition_completed):
                    raise ValueError('strict collision cannot publish ordinary negative evidence')
                if p.target_transition_completed and last_scan != key:
                    raise ValueError('missing scan before completed target transition')
                group[1] = event
                completed = key
                found, terminated = p.found_after, p.episode_terminated
            else:
                raise ValueError('public event kind/payload mismatch')
            cursor, step = event.sequence, key
            seen[event.event_id] = event.payload_sha256
            while len(seen) > self.capacity:
                seen.popitem(last=False)
        self.episode_id, self.cursor, self.physical_step, self.completed_step = episode, cursor, step, completed
        self._seen = seen
        self.last_scan_step, self.found, self.terminated = last_scan, found, terminated
        return tuple(PublicStepEvidence(episode, key, tuple(value[0]), value[1]) for key, value in groups.items())
