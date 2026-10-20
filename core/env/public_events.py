"""Optional immutable copies of observations already used by the environment.

This module contains no inference or planning. A bounded journal never calls
consumer code, changes physical state, or queries the hidden world.
"""
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json


def vectors(values):
    return tuple(tuple(float(x) for x in row) for row in values)


@dataclass(frozen=True)
class ScanObservation:
    origins: tuple
    directions: tuple
    distances: tuple
    hit_mask: tuple
    agent_ids: tuple


@dataclass(frozen=True)
class StepObservation:
    agent_start: tuple
    agent_end: tuple
    velocity_start: tuple
    velocity_end: tuple
    found_before: bool
    found_after: bool
    target_transition_completed: bool
    discovery_check_completed: bool
    episode_terminated: bool
    terminal_reason: str

    @property
    def negative_observation_valid(self):
        return (self.target_transition_completed and self.discovery_check_completed
                and not self.found_before and not self.found_after)


@dataclass(frozen=True)
class PublicEvent:
    episode_id: str
    physical_step: int
    sequence: int
    source_id: str
    kind: str
    payload: ScanObservation | StepObservation
    payload_sha256: str

    @property
    def event_id(self):
        return (self.episode_id, self.physical_step, self.source_id, self.kind, self.sequence)


def payload_digest(payload):
    data = json.dumps(asdict(payload), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class PublicEventJournal:
    def __init__(self, episode_id, *, capacity=1024):
        if not isinstance(episode_id, (str, int)) or isinstance(episode_id, bool) or not str(episode_id):
            raise ValueError('a nonempty public episode identifier is required')
        if type(capacity) is not int or not 2 <= capacity <= 8192:
            raise ValueError('public event capacity must be in [2,8192]')
        self.label = str(episode_id)
        self.capacity = capacity
        self.generation = 0
        self.episode_id = ''
        self.sequence = 0
        self.events = deque(maxlen=capacity)

    def reset(self):
        self.generation += 1
        self.episode_id = f'{self.label}:{self.generation}'
        self.sequence = 0
        self.events.clear()

    def append(self, step, source, kind, payload):
        self.sequence += 1
        event = PublicEvent(self.episode_id, int(step), self.sequence, str(source),
                            str(kind), payload, payload_digest(payload))
        self.events.append(event)
        return event

    def read(self, after_sequence=0):
        if type(after_sequence) is not int or not 0 <= after_sequence <= self.sequence:
            raise ValueError('invalid public event cursor')
        if self.events and after_sequence < self.events[0].sequence - 1:
            raise ValueError('public event history was evicted')
        return tuple(event for event in self.events if event.sequence > after_sequence)
