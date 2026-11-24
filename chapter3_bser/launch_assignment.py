"""Local departure offers on top of the frozen neighbor navigation runtime."""
import copy
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
import torch

from chapter3_bser.neighbor_ctde.mission import NeighborMission
from chapter3_bser.r123.options import CONTROL
from chapter3_bser.r123_task_sync.receipts import snapshot
from chapter3_bser.safety.planning_refresh import PlanningViews


@dataclass(frozen=True)
class DepartureMenu:
    """The selector sees its own role, current received observation and feasible points in draw order."""
    boat: int
    step: int
    features: tuple
    points: tuple


@dataclass(frozen=True)
class OfferDraw:
    stream: str
    latent: tuple
    raw_goal: tuple
    projected: tuple | None
    option: int | None
    status: str
    queries: int


def first_feasible(menu):
    return 0 if menu.points else None


def _check_private(streams):
    if any(not isinstance(g, torch.Generator) or g is torch.default_generator
           or g.device.type != 'cpu' for g in streams):
        raise ValueError('expected private CPU random streams')
    if len({id(g) for g in streams}) != len(streams):
        raise ValueError('random streams must be independent objects')


class DepartureBoat:
    """Only the step-zero request of this addressed search boat uses the menu."""

    def __init__(self, boat, original, extra, choose):
        if boat.agent not in range(3):
            raise ValueError('only search boats have departure menus')
        self.boat = boat
        self.original, self.extra, self.choose = original, extra, choose
        self._menu = None
        self._draws = ()
        self._state = None
        self._preview_queries = 0
        self._admission_interrupted = False

    def __getattr__(self, name):
        return getattr(self.boat, name)

    @property
    def proposal_queries(self):
        return self._preview_queries + self.boat.proposal_queries

    def _open(self):
        if self._admission_interrupted:
            raise RuntimeError('native admission was interrupted; replay from a fresh mission')
        nav = self.boat.navigator
        frame = nav.env.frame
        return (frame.step == 0 and not frame.stopped and nav.agent not in nav.lifecycle.tasks
                and frame.step - nav._last_request >= CONTROL['request_interval'])

    def _features(self):
        nav = self.boat.navigator
        if any(r.step < 0 for r in self.boat.inbox.latest.values()):
            raise ValueError('report predates the mission clock')
        context = self.boat.evidence.context(nav, self.boat.inbox)
        return tuple(np.concatenate((nav.features, context)).tolist())

    def preview(self, actor):
        """Seal one menu. Re-reading it consumes no draws, queries or mission steps."""
        if not self._open():
            raise ValueError('no open departure request')
        nav = self.boat.navigator
        features = self._features()
        if self._menu is not None:
            if self._state is not nav.state or features != self._menu.features:
                raise ValueError('departure observation changed after preview')
            return self._menu

        # Native _project reads the onboard map. Only its diagnostics and its
        # filtered-graph cache are mutable. Keep both private to this preview.
        probe = copy.copy(nav)
        probe.planning_views = PlanningViews(probe, nav.planning_views.clearance)
        probe.proposal_queries = 0
        streams = [('original_offer', self.original)]
        if self.extra is not None:
            streams.append(('extra_offer', self.extra))
        # Stage RNG state too: an interrupted projection must not redraw a
        # different prefix on retry. Count even work discarded by an exception.
        staged = [(name, rng.clone_state()) for name, rng in streams]
        draws, points = [], []
        try:
            for name, rng in staged:
                latent, unit = actor.propose(features, rng)
                raw = (unit + 1.) * .5 * nav.bounds
                before = probe.proposal_queries
                projected = probe._project(raw)
                point = None if projected is None else tuple(projected.tolist())
                option = None
                if point is None:
                    status = 'infeasible'
                elif point in points:
                    status, option = 'duplicate', points.index(point)
                else:
                    status, option = 'feasible', len(points)
                    points.append(point)
                draws.append(OfferDraw(name, tuple(latent.tolist()), tuple(raw.tolist()),
                                       point, option, status, probe.proposal_queries - before))
        finally:
            self._preview_queries += probe.proposal_queries
        for (_, rng), (_, sampled) in zip(streams, staged):
            rng.set_state(sampled.get_state())
        self._draws = tuple(draws)
        self._state = nav.state
        self._menu = DepartureMenu(nav.agent, nav.env.frame.step, features, tuple(points))
        return self._menu

    def submit(self, menu, option):
        """Consume the exact preview once, then read back the native admission and route."""
        nav = self.boat.navigator
        if menu is None or menu is not self._menu or self._state is not nav.state or not self._open():
            raise ValueError('stale, foreign or consumed departure menu')
        if menu.features != self._features():
            raise ValueError('departure observation changed before submission')
        if menu.points:
            if isinstance(option, bool) or not isinstance(option, Integral) or option not in range(len(menu.points)):
                raise ValueError('select one feasible option')
            option = int(option)
        elif option is not None:
            raise ValueError('select one feasible option, or None for an empty menu')
        chosen = next((draw for draw in self._draws
                       if draw.status == 'feasible' and draw.option == option), None)
        # These are the two writes made by LocalNavigator.act on an eligible
        # proposal. Consuming its request interval lets the same act() perform
        # native admission, tracking and safety once, without projecting again.
        self._admission_interrupted = True
        nav.point = None if chosen is None else np.array(chosen.projected)
        nav._last_request = menu.step
        command, assignment, unexpected = nav.act(None)
        if unexpected is not None:
            raise RuntimeError('native submission requested another proposal')
        receipt = snapshot(nav, nav.state)[nav.agent]
        task = nav.lifecycle.tasks[nav.agent]
        accepted = bool(chosen is not None and not task.released
                        and np.allclose(chosen.projected, task.point, rtol=0, atol=1e-6))
        follows = bool(accepted and receipt.status == 'ACTIVE' and receipt.reachable and assignment.planned_path
                       and np.allclose(assignment.planned_path[-1], chosen.projected, rtol=0, atol=1e-6))
        if chosen is None:
            fallback = 'no_feasible_offer'
        elif not accepted:
            fallback = 'native_rejection'
        elif not follows:
            fallback = 'native_route_differs_or_unreachable'
        else:
            fallback = None
        # A fresh mission has no peer reports at t=0. SharedEvidence rejects
        # a same-step or future report before any candidate can be drawn.
        decision = dict(agent=nav.agent, step=menu.step, features=list(menu.features),
                        peer_reports={}, inbox_effect_m=0.,
                        latent=None if chosen is None else list(chosen.latent),
                        raw_goal=None if chosen is None else list(chosen.raw_goal),
                        proposed=None if chosen is None else list(chosen.projected),
                        accepted=accepted, root=list(receipt.root), route_id=receipt.route_id,
                        departure=dict(draws=[asdict(d) for d in self._draws], options=menu.points,
                                       selected_option=option, selected_stream=chosen.stream if chosen else None,
                                       preview_queries=self._preview_queries, task_id=receipt.task_id,
                                       admission_status=receipt.status, release_reason=task.release_reason,
                                       route_reachable=receipt.reachable, route_to_selected=follows,
                                       route_points=tuple(tuple(map(float, p)) for p in assignment.planned_path),
                                       fallback=fallback))
        self._admission_interrupted = False
        return command, assignment, decision

    def act(self, actor, generator=None):
        if not self._open():
            return self.boat.act(actor, generator)
        menu = self.preview(actor)
        option = self.choose(menu) if len(menu.points) > 1 else first_feasible(menu)
        return self.submit(menu, option)


class LaunchAssignmentMission(NeighborMission):
    """Native neighbor mission with one or two private offers at departure."""

    def __init__(self, config, options, scene, *, radius, originals, extras=None, selector_factory=None):
        if len(originals) != 3 or (extras is not None and len(extras) != 3):
            raise ValueError('expected one stream per search boat for each enabled draw')
        self._offer_streams = tuple(originals) + (() if extras is None else tuple(extras))
        _check_private(self._offer_streams)
        selectors = [first_feasible] * 3
        if selector_factory is not None:
            selectors = [selector_factory() for _ in range(3)]
            if any(not callable(rule) for rule in selectors) or len({id(rule) for rule in selectors}) != 3:
                raise ValueError('create a separate onboard selector for each search boat')
        super().__init__(config, options, scene, radius=radius)
        for i in range(3):
            self.boats[i] = DepartureBoat(self.boats[i], originals[i],
                                          None if extras is None else extras[i], selectors[i])

    def rollout(self, actor, generators, cap, *, log_training=False, departure=None, observe=None):
        if log_training:
            raise ValueError('a selected departure is not an on-policy draw for the legacy trainer')
        if len(generators) != 3:
            raise ValueError('expected three continuation streams')
        _check_private((*self._offer_streams, *generators))
        return super().rollout(actor, generators, cap, departure=departure, observe=observe)
