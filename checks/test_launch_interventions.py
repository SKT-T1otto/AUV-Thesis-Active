from dataclasses import dataclass, replace
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from chapter3_bser.joint_departure import JointDeparture
from chapter3_bser.local_ctde.mission import pack_guidance
from chapter3_bser.integration.control_context import AgentAssignmentContextV1
from tools.launch_assignment_audit import admissions, labels


class LabelTests(unittest.TestCase):
    xyz = [0., 0., 0.]
    far = [5., 0., 0.]

    def trace(self, steps, horizon, *, winner=1, tie=False, collision=False, success=False):
        p = [[0. if tie else .4, 0., 0.], self.xyz, self.far, self.xyz]
        initial = dict(kinematics=dict(positions=p), target=dict(position=self.xyz))
        protocol = dict(horizon=horizon, radius=1., hold=5, ranges=[1.]*3, bias=0.)
        rows = []
        for t in range(1, steps+1):
            crash = collision and t == steps
            complete = success and t == 6
            reason = 'obstacle_collision' if crash else 'success' if complete else 'timeout' if t == horizon else 'running'
            target = dict(position=self.xyz, velocity=self.xyz)
            packet = dict(physical_step=t, received_step=2, payload=dict(**target, sample_step=1)) if t >= 2 else None
            row = dict(step=t, collision=crash, target=target, finder=winner, packet=packet, native_success=complete,
                       event=dict(agent_start=p, agent_end=p, found_before=t>1, found_after=True,
                                  target_transition_completed=not crash, discovery_check_completed=t==1 and not crash,
                                  terminal_reason=reason, episode_terminated=reason!='running'),
                       team=float(t), broadcast=[float(t)]*4, source_rewards=[float(t)]*4)
            rows.append(row)
        return initial, rows, protocol, steps

    def test_same_step_hits_keep_all_and_register_by_distance_then_id(self):
        for tie, expected in ((False,1),(True,0)):
            result = labels(*self.trace(1, 3, winner=expected, tie=tie))
            self.assertEqual(result['same_step_hits'], [0,1])
            self.assertEqual(result['registered_finder'], expected)
        with self.assertRaisesRegex(ValueError, 'registration'):
            labels(*self.trace(1, 3, winner=0))

    def test_timeout_on_discovery_has_no_receipt(self):
        r = labels(*self.trace(1, 1))
        self.assertEqual((r['found_step'],r['receipt_step'],r['S'],r['reason']), (1,None,False,'timeout'))

    def test_packet_read_on_final_or_collision_step(self):
        for crash in (False,True):
            args = self.trace(2, 2, collision=crash)
            r = labels(*args)
            self.assertEqual((r['receipt_step'],r['S']), (2,False))
            args[1][-1]['packet'] = None
            with self.assertRaisesRegex(ValueError, 'receipt'):
                labels(*args)

    def test_censoring_and_discount_start_at_zero(self):
        r = labels(*self.trace(2, 10))
        self.assertIsNone(r['S'])
        self.assertIsNone(r['terminal_discounted_return'])
        self.assertAlmostEqual(r['observed_discounted_return'], 1 + .95*2)

    def test_five_full_postreceipt_intervals_and_false_success(self):
        r = labels(*self.trace(6, 10, success=True))
        self.assertEqual((r['S'], r['success_step']), (True,6))
        args = self.trace(3, 10)
        args[1][-1]['native_success'] = True
        with self.assertRaisesRegex(ValueError, 'success'):
            labels(*args)

    def test_collision_before_sensing_yields_known_failure_without_discovery(self):
        args = self.trace(1, 10, collision=True)
        args[1][0].update(finder=-1)
        args[1][0]['event']['found_after'] = False
        result = labels(*args)
        self.assertEqual((result['S'], result['found_step'], result['same_step_hits']), (False,None,[]))


@dataclass
class Menu:
    points: tuple


class BarrierTests(unittest.TestCase):
    def test_native_float32_waypoint_has_the_same_guidance_token(self):
        point = (.5, 1.25, 2.)
        task = SimpleNamespace(mission_complete=False, target_found=False, step=27)
        items = [AgentAssignmentContextV1(i, 'executor' if i==3 else 'searcher', 'standby' if i==3 else 'search',
                 str(i), point, (point,), point, point, False, True, False) for i in range(4)]
        reference = pack_guidance(items, task)
        items[2] = replace(items[2], final_waypoint=tuple(np.asarray(point, dtype=np.float32)))
        self.assertEqual(pack_guidance(items, task).allocation_hash, reference.allocation_hash)

    def setup_boats(self):
        events = []
        class Boat:
            def __init__(self, i):
                self.i = i
                self.env = SimpleNamespace(frame=SimpleNamespace(step=0))
                self.original, self.extra = [torch.Generator().manual_seed(i+n) for n in (0, 5)]
                self._draws = []
            def preview(self, policy):
                events.append(('preview', self.i))
                return Menu(((0,), (1,)))
            def submit(self, menu, option):
                self.assert_barrier()
                events.append(('submit', self.i))
                return self.i, option
            def assert_barrier(self):
                if sorted(i for op,i in events if op=='preview') != [0,1,2]:
                    raise AssertionError('submission before every preview')
            def act(self, policy):
                self.assert_barrier()
                events.append(('executor', self.i))
                return self.i, None
        return [Boat(i) for i in range(4)], events

    def test_barrier_reverse_order_canonical_slots_and_one_use(self):
        results = []
        for order in (range(4), reversed(range(4))):
            boats, events = self.setup_boats()
            t = JointDeparture([1,0,0], order)
            args=(boats,[None]*3,[b.original.clone_state() for b in boats[:3]])
            results.append(t(*args))
            self.assertEqual(len(events), 7)
            with self.assertRaisesRegex(ValueError, 'consumed'):
                t(*args)
        self.assertEqual(results, [[(0,1),(1,0),(2,0),(3,None)]]*2)

    def test_invalid_assignment_cannot_partially_submit(self):
        boats, events = self.setup_boats()
        t = JointDeparture([0,0,2])
        with self.assertRaisesRegex(ValueError, 'outside'):
            t(boats, [None]*3, [b.original.clone_state() for b in boats[:3]])
        self.assertEqual([op for op,_ in events], ['preview']*3)
        self.assertFalse(t.consumed)


class ReceiptTests(unittest.TestCase):
    def test_rejected_and_rerouted_assignments_stay_in_the_original_arm(self):
        roots, routes, reports, decisions, menus, trials = [], [], [], [], [], []
        for i in range(3):
            point, other = [1., 2., 3.], [2., 3., 4.]
            chosen = 1 if i == 0 else 0
            proposed = [point, other][chosen]
            root = [0., 0., 0.] if i == 0 else point
            status = ('RELEASED','UNREACHABLE','ACTIVE')[i]
            route = f'route-{i}'
            draws = [dict(stream=name, option=k, projected=p, status='feasible')
                     for k,(name,p) in enumerate(zip(('original_offer','extra_offer'),(point,other)))]
            receipt = dict(draws=draws, options=[point,other], selected_option=chosen,
                selected_stream=draws[chosen]['stream'], task_id=int(i!=0), admission_status=status,
                route_reachable=i==2, route_points=[root], route_to_selected=i==2,
                fallback=('native_rejection','native_route_differs_or_unreachable',None)[i])
            roots.append(root)
            routes.append(route)
            reports.append(dict(step=0, origin=i, task_id=int(i!=0), active=i!=0,
                                root=root if i!=0 else None, remaining_route=[root]))
            decisions.append(dict(step=0, agent=i, features=[], departure=receipt, proposed=proposed,
                                  root=root, route_id=route, accepted=i!=0))
            menus.append(dict(step=0, boat=i, features=[], points=[point,other]))
            trials.append(draws)
        branch = dict(assignment=[1,0,0], prefix=dict(menus=menus, draws=trials,
            streams_before=[['a','b','c']]*3, streams_after=[['x','y','c']]*3),
            episode=dict(decisions=decisions, frames=[dict(roots=roots,routes=routes)],
                         communication=dict(reports=reports)))
        rows = admissions(branch)
        self.assertEqual([r['assigned'] for r in rows], [1,0,0])
        self.assertEqual([r['accepted'] for r in rows], [False,True,True])
        decisions[0]['accepted'] = True
        with self.assertRaisesRegex(ValueError, 'acceptance'):
            admissions(branch)
