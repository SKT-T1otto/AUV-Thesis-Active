"""One offline assignment at the departure barrier. Returns actions in simulator slot order."""
from dataclasses import asdict
import hashlib


def stream_state(g):
    return hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest()


def stream_states(boats, continuation):
    return [[stream_state(g) for g in (b.original, b.extra, continuation[i])]
            for i, b in enumerate(boats[:3])]


class JointDeparture:
    def __init__(self, assignment, order=(0, 1, 2, 3)):
        self.assignment, self.order = tuple(assignment), tuple(order)
        if len(self.assignment) != 3 or any(type(i) is not int for i in self.order) or sorted(self.order) != list(range(4)):
            raise ValueError('expected three assignments and a permutation of four boats')
        self.prefix = None
        self.consumed = False

    def __call__(self, boats, policies, continuation):
        if self.consumed or any(b.env.frame.step != 0 for b in boats):
            raise ValueError('departure ticket already consumed or outside step zero')
        before = stream_states(boats, continuation)
        menus = [None] * 3
        for i in self.order:
            if i < 3:
                menus[i] = boats[i].preview(policies[i])
        self.prefix = dict(streams_before=before, streams_after=stream_states(boats, continuation),
                           menus=[asdict(menu) for menu in menus],
                           draws=[[asdict(draw) for draw in b._draws] for b in boats[:3]])
        for menu, choice in zip(menus, self.assignment):
            if (menu.points and (type(choice) is not int or choice not in range(len(menu.points)))) or (
                    not menu.points and choice is not None):
                raise ValueError('assignment outside the frozen menu')
        self.consumed = True  # a partial native submission must abort this mission
        actions = [None] * 4
        for i in self.order:
            actions[i] = boats[i].submit(menus[i], self.assignment[i]) if i < 3 else boats[i].act(None)
        return actions
