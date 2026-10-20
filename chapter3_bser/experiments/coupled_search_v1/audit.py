"""Observation-only audit; no allocation, reward, or physical-state mutation."""
from collections import Counter, deque
import hashlib
import json
from .public_observations import PublicObservationAdapter


def installed_route_hash(context):
    # Moving hold_position fields are irrelevant while following a route.
    payload = [dict(agent_id=item.agent_id, role=item.role, assignment_id=item.assignment_id,
                    final=item.final_waypoint, path=item.planned_path,
                    hold=item.hold_state, reachable=item.reachable,
                    hold_position=item.hold_position if item.hold_state else None,
                    execution_request=item.execution_request) for item in context.agent_assignments]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class PublicEvidenceSession:
    def __init__(self, config, vehicle_specs, control_contract):
        self.config = config.copy()
        self.vehicle_specs = vehicle_specs
        self.control_contract = control_contract
        self.adapter = PublicObservationAdapter(capacity=config['event_capacity'])
        self.counts = Counter()
        self.records = deque(maxlen=config['audit_capacity'])
        self.last_route_hash = None
        self.valid = True
        self.error = None

    def consume(self, events):
        if not self.valid:
            return ()
        try:
            evidence = self.adapter.ingest(events)
        except ValueError as error:
            self.valid, self.error = False, str(error)
            self.counts['invalid_evidence_batches'] += 1
            return ()
        for item in evidence:
            self.counts['scan_batches'] += len(item.scans)
            self.counts['ray_measurements'] += sum(len(e.payload.origins)*len(e.payload.directions) for e in item.scans)
            if item.transition:
                self.counts['physical_transitions'] += 1
                self.counts['negative_observations'] += int(item.negative_observation_valid)
                self.counts['discovery_checks'] += int(item.transition.payload.discovery_check_completed)
                self.counts['terminal_transitions'] += int(item.transition.payload.episode_terminated)
            self.records.append(dict(kind='public_evidence', step=item.physical_step,
                                     scan_event_ids=[e.event_id for e in item.scans],
                                     transition_event_id=None if item.transition is None else item.transition.event_id,
                                     negative_observation_valid=item.negative_observation_valid))
        return evidence

    def installed(self, context):
        digest = installed_route_hash(context)
        if digest != self.last_route_hash:
            self.records.append(dict(kind='installed_route', step=int(context.step), route_sha256=digest,
                                     previous_route_sha256=self.last_route_hash,
                                     hold_agent_ids=[x.agent_id for x in context.agent_assignments if x.hold_state],
                                     reason=context.decision_reason))
            self.counts['route_versions'] += 1
        self.last_route_hash = digest

    def summary(self):
        return dict(schema='ch3.coupled_search.public_evidence.v1', mode='observe_only',
                    search_planner_id='d2_v1', algorithm_changed=False,
                    estimator_implemented=False, valid=self.valid, error=self.error,
                    last_event_sequence=self.adapter.cursor, counts=dict(self.counts),
                    retained_audit_records=len(self.records), latest_route_sha256=self.last_route_hash)
