"""Detached public decision inputs and factual labels for one mission prefix."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import isfinite

import numpy as np

from core.mapping.planning_state import planning_state_sha256


SCHEMA = "auv.handoff_value.records.v1"


def _route(candidate):
    points = np.asarray(candidate.path_points, dtype="<f8").reshape(-1, 3)
    route = dict(candidate_id=candidate.candidate_id, waypoint=list(candidate.waypoint),
                 points=points.tolist(), polyline_sha256=sha256(points.tobytes()).hexdigest(),
                 cell_indices=candidate.path_cell_indices.tolist(), source=candidate.source)
    for name in ("path_length", "planning_cost", "physical_travel_time"):
        value = float(getattr(candidate, name))
        route[name] = value if isfinite(value) else None
    return route


class EpisodeRecords:
    def __init__(self, scene_id, episode_id):
        self.identity = dict(scene_id=scene_id, episode_id=int(episode_id))
        self.decisions = []
        self.first_handoff = None
        self.team_rewards = []

    def record_proposal(self, physical_step, state, candidates, frozen, standbys, result,
                        trigger, receipts, *, initializing):
        """Observe the actual solver arguments in order and return its original object."""
        step = int(physical_step)
        previous = self.decisions[-1] if self.decisions else None
        call_index = previous["call_index"] + 1 if previous and previous["step"] == step else 0
        fixed = []
        for c in frozen:
            receipt = receipts.get(c.agent_id)
            fixed.append(dict(agent=int(c.agent_id), route=_route(c),
                              receipt=asdict(receipt) if receipt is not None else None))
        self.decisions.append(dict(
            **self.identity, feature_version=SCHEMA, step=step, call_index=call_index,
            planning_step=int(state.step), map_revision=int(state.map_revision),
            belief_revision=int(state.target_belief.revision),
            public_state_sha256=planning_state_sha256(state),
            stage="initialization" if initializing else "search", trigger=trigger,
            agents=[dict(agent=int(a.agent_id), position=list(a.position), velocity=list(a.velocity))
                    for a in state.agents],
            candidates=[dict(agent=int(c.agent_id), **_route(c)) for c in candidates],
            fixed=fixed, standbys=[_route(c) for c in standbys],
            native_selected=[dict(agent=int(c.agent_id), candidate_id=c.candidate_id,
                                  waypoint=list(c.waypoint)) for c in result.selected],
            native_standby=dict(candidate_id=result.standby.candidate_id,
                                waypoint=list(result.standby.waypoint)),
            native_objective=float(result.objective), native_status=result.status,
        ))
        return result

    def end_step(self, step, team_reward, env):
        """Sample the public delivery at every completed interval, including termination."""
        if step != len(self.team_rewards) + 1:
            raise ValueError("expected one public packet read per consecutive physical step")
        packet = env.get_public_delivered_target()
        if packet is not None and self.first_handoff is None:
            if packet["physical_step"] != step:
                raise ValueError("public packet read on a different physical step")
            agents = env.get_agent_state()
            executor = env.get_search_execution_state().executor_id
            self.first_handoff = dict(
                **self.identity, read_step=step, received_step=int(packet["received_step"]),
                packet_schema=packet["schema"], payload=deepcopy(packet["payload"]),
                executor_id=int(executor), position=list(agents.positions[executor]),
                velocity=list(agents.velocities[executor]),
            )
        self.team_rewards.append(float(team_reward))

    def outcome(self, task, episode, found_step):
        """A cut prefix can establish an observed event, but leaves the final label open."""
        ended = bool(episode["episode_terminated"])
        reason = episode["termination_reason"] if ended else None
        found = bool(task.target_found)
        delivered = self.first_handoff is not None
        success = bool(episode["success"]) if ended else None
        return dict(
            **self.identity, observed_through_step=int(task.step), censored=not ended,
            terminal_label=("success" if success else "failure") if ended else "unknown",
            termination_reason=reason, terminal_step=episode["terminal_step"] if ended else None,
            found=found if ended or found else None, found_step=found_step,
            handoff=delivered if ended or delivered else None, handoff_event_step=task.handoff_step,
            received_step=self.first_handoff["received_step"] if delivered else None,
            success=success, collision=bool(episode["collision_episode"]) if ended else None,
            timeout=(reason == "timeout") if ended else None,
        )

    def export(self, task, episode, found_step):
        return dict(schema=SCHEMA, decisions=deepcopy(self.decisions), first_handoff=deepcopy(self.first_handoff),
                    packet_reads=len(self.team_rewards), team_rewards=list(self.team_rewards),
                    outcome=self.outcome(task, episode, found_step))
