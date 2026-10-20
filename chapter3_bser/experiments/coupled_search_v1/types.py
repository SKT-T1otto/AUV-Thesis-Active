from dataclasses import dataclass
from core.env.public_events import PublicEvent
import numpy as np


@dataclass(frozen=True)
class PublicStepEvidence:
    episode_id: str
    physical_step: int
    scans: tuple[PublicEvent, ...]
    transition: PublicEvent | None

    @property
    def negative_observation_valid(self):
        return bool(self.transition and self.transition.payload.negative_observation_valid)


@dataclass(frozen=True)
class TargetForecast:
    positions: np.ndarray
    weights: np.ndarray
    map_ids: tuple
    step: int
    identity: str


@dataclass(frozen=True)
class RouteProposal:
    candidate: object
    tracker_index: int = 0
    reference: bool = False
    blocked: bool = False

    @property
    def agent_id(self):
        return self.candidate.agent_id

    @property
    def route_id(self):
        return self.candidate.candidate_id


@dataclass(frozen=True)
class ExecutionForecast:
    positions: np.ndarray
    valid_steps: np.ndarray
    truncation_reasons: tuple
    identity: str


@dataclass(frozen=True)
class DiscoverySignature:
    values: np.ndarray
    first_steps: np.ndarray
    target_identity: str
    execution_identity: str


@dataclass(frozen=True)
class CompressionResult:
    indices: tuple
    epsilon: float
    reached_error_target: bool
    stopped_by: str
