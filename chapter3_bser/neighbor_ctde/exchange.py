"""Reliable one-step, range-limited task-data exchange before handoff."""
from collections import defaultdict
from dataclasses import dataclass
from math import dist, isfinite


@dataclass(frozen=True)
class Report:
    origin: int
    step: int
    position: tuple
    velocity: tuple
    task_id: int
    active: bool
    root: tuple | None
    remaining_route: tuple
    scans: tuple
    checked_clear: bool

    @property
    def key(self):
        return self.origin, self.step


class Inbox:
    """Facts retain their original observer and acquisition time across relays."""

    def __init__(self, agent):
        self.agent = agent
        self.records = {}
        self.latest = {}

    def add(self, report):
        if report.key in self.records:
            if self.records[report.key] != report:
                raise ValueError('conflicting versions of one observer/step report')
            return False
        self.records[report.key] = report
        previous = self.latest.get(report.origin)
        if previous is None or report.step > previous.step:
            self.latest[report.origin] = report
        return True


class RangeExchange:
    """Take a send-time snapshot before any delivery, at most one hop per tick.

    Each sender remembers which fact IDs it has sent on each reliable link.
    This gives a newly encountered neighbor the full available history while
    avoiding repeated transmission of unaltered facts on a persistent link.
    """

    def __init__(self, radius):
        if not isfinite(radius) or radius < 0:
            raise ValueError('communication radius must be finite and nonnegative')
        self.radius = radius
        self.sent = defaultdict(set)
        self.trace = []
        self.last_send_step = -1

    def deliver(self, step, reports, inboxes):
        if step != self.last_send_step + 1:
            raise ValueError('nonconsecutive radio send step')
        if len(reports) != 4 or len(inboxes) != 4:
            raise ValueError('expected the four on-board participants')
        self.last_send_step = step
        for i, report in enumerate(reports):
            if inboxes[i].agent != i:
                raise ValueError('mailbox identity mismatch')
            if report is not None:
                if report.key != (i, step):
                    raise ValueError('origin or send clock mismatch')
                inboxes[i].add(report)
        batches = []
        for src, tx in enumerate(reports):
            if tx is None:
                continue
            for dst, rx in enumerate(reports):
                if src == dst or rx is None or dist(tx.position, rx.position) > self.radius:
                    continue
                sent = self.sent[src, dst]
                keys = sorted(inboxes[src].records.keys() - sent)
                if keys:
                    batches.append((src, dst, tuple(inboxes[src].records[k] for k in keys)))
                    sent.update(keys)
        arrivals = [[] for _ in reports]
        for src, dst, payload in batches:
            added = []
            for report in payload:
                if report.step > step:
                    raise ValueError('future report on search coordination channel')
                if inboxes[dst].add(report):
                    arrivals[dst].append(report)
                    added.append(report.key)
            self.trace.append(dict(send=step, receive=step + 1, sender=src, receiver=dst,
                distance=dist(reports[src].position, reports[dst].position),
                ids=[r.key for r in payload], first_received=added))
        return arrivals
