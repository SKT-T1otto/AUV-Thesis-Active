"""Four on-board controllers exchanging only previously observed search facts."""
import numpy as np
import torch

from chapter3_bser.local_ctde.mission import LocalMission
from chapter3_bser.local_ctde.observation import sensor_frame
from .exchange import Inbox, RangeExchange, Report
from .observations import SharedEvidence
from .policy import ReceivedPolicy


class NeighborBoat:
    """Add an addressed inbox to one existing on-board navigator."""

    def __init__(self, navigator):
        self.navigator = navigator
        self.inbox = Inbox(navigator.agent)
        self.evidence = SharedEvidence(navigator.env)
        self.checked_clear = False

    def __getattr__(self, name):
        return getattr(self.navigator, name)

    def report(self):
        i, frame = self.agent, self.env.frame
        # The public handoff remains the only way the executor acquires a target.
        if frame.stopped or frame.delivery is not None:
            return None
        task = self.lifecycle.tasks[i]
        c = self.manager.current[i]
        active = not task.released and task.identity > 0
        route = ()
        if active and c.route.reachable:
            route = tuple(tuple(float(x) for x in row) for row in c.route.points[c.route.index:])
        return Report(i, frame.step, frame.position, frame.velocity, int(task.identity), bool(active),
                      tuple(map(float, task.point)) if active else None, route, frame.scans,
                      self.checked_clear)

    def sense_delivered(self, frame, arrivals, checked):
        self.evidence.before_sensing(self.env, frame, arrivals)
        self.navigator.sense(frame)
        self.checked_clear = bool(self.agent < 3 and checked and not frame.stopped)

    def act(self, actor, generator=None):
        if self.agent == 3:
            return self.navigator.act(None)
        context = self.evidence.context(self.navigator, self.inbox)
        cmd, assignment, draw = self.navigator.act(ReceivedPolicy(actor, context), generator)
        if draw is not None:
            own = np.asarray(draw['features'], dtype=np.float32)
            draw['features'] = np.concatenate((own, context)).tolist()
            draw['peer_reports'] = {str(j): entry.step for j, entry in self.inbox.latest.items()
                                    if j != self.agent}
            # Same own state/weights and same latent noise; remove the received
            # policy input only. This is a raw-goal diagnostic, not a second rollout.
            full_mean, _ = actor.propose(draw['features'])
            muted_mean, _ = actor.propose(np.concatenate((own, np.zeros_like(context))))
            muted_z = np.asarray(draw['latent'], dtype=np.float32) - (full_mean - muted_mean)
            muted_goal = .5 * (1. + torch.from_numpy(muted_z).tanh().numpy()) * self.bounds
            draw['inbox_effect_m'] = (float(np.linalg.norm(np.asarray(draw['raw_goal']) - muted_goal))
                                      if draw['peer_reports'] else 0.)
        return cmd, assignment, draw


class NeighborMission(LocalMission):
    def __init__(self, config, options, scene, *, radius):
        self.radio = RangeExchange(radius)
        super().__init__(config, options, scene)

    def receive(self):
        task = self.core.get_task_state()
        if task.step == 0:
            self.boats = [NeighborBoat(boat) for boat in self.boats]
        arrivals = [[] for _ in self.boats]
        if task.step > 0:
            # All local actions for t already completed. Reports and link geometry
            # refer to the stored start-of-t sensor frames, never t+1 positions.
            arrivals = self.radio.deliver(task.step - 1, [b.report() for b in self.boats],
                                           [b.inbox for b in self.boats])
        kin = self.core.get_agent_state()
        finished = self.core.get_search_execution_state().agent_finished
        events = self.core.read_public_events(self.cursor)
        checked = any(e.kind == 'step' and e.physical_step == task.step and
                      e.payload.negative_observation_valid for e in events)
        delivery = self.core.get_public_delivered_target()
        for i, boat in enumerate(self.boats):
            frame = sensor_frame(i, task.step, kin, finished, events, delivery)
            boat.sense_delivered(frame, arrivals[i], checked)
        if events:
            self.cursor = events[-1].sequence
        return task, delivery

    def rollout(self, *args, **kwargs):
        result = super().rollout(*args, **kwargs)
        facts = [dict(origin=r.origin, step=r.step, position=r.position, velocity=r.velocity,
                      task_id=r.task_id, root=r.root, active=r.active, remaining_route=r.remaining_route,
                      scan_count=len(r.scans), checked_clear=r.checked_clear)
                 for boat in self.boats for r in boat.inbox.records.values() if r.origin == boat.agent]
        result['communication'] = dict(radius=self.radio.radius, reports=facts,
            packets=len(self.radio.trace), links=self.radio.trace,
            received_scans=[b.evidence.received_scans for b in self.boats],
            received_checks=[b.evidence.received_checks for b in self.boats],
            informed_draws=sum(bool(d['peer_reports']) for d in result['decisions']))
        return result
