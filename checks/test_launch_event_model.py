import unittest
from fractions import Fraction as F
from itertools import permutations, product

from chapter3_bser.handoff_value.event_model import (
    Window, contrast_bounds, event_rows, fourth_arm_interval, standby_contrasts,
)
from tools import launch_event_study as study


ARMS = list(product(range(2), range(2), range(1)))


def known(rows, orders, certificates=None):
    windows = [[Window(t,t) for t in menu] for menu in orders]
    cert = {a:'same continuation law' for a in rows} if certificates is None else certificates
    return event_rows(windows, cert, horizon=20, hold=3)


class EventTests(unittest.TestCase):
    def test_three_boat_all_720_total_orders_against_direct_first_event(self):
        arms = list(product(range(2), repeat=3))
        largest = 0
        for order in permutations(range(1,7)):
            times = [order[i:i+2] for i in (0,2,4)]
            rows = known(arms, times)
            direct = {a:min(times[i][v] for i,v in enumerate(a)) for a in arms}
            for a,b in product(arms, repeat=2):
                self.assertEqual(rows[a] == rows[b], direct[a] == direct[b])
            largest = max(largest, len(set(rows.values())))
        self.assertEqual(largest, 4)

    def test_mixed_registration_orders_imply_fourth_mean(self):
        rows = [known(ARMS, [[1,2],[4,5],[9]]), known(ARMS, [[4,5],[1,2],[9]])]
        # Independent oracle: class 1 uses only boat 0, class 2 only boat 1.
        u,v = (F(1,5),F(4,5)), (F(1,10),F(1,2))
        q = {a:(u[a[0]]+v[a[1]])/2 for a in ARMS}
        self.assertEqual(len(set(q.values())),4)
        measured = {a:(q[a],q[a]) for a in ARMS[:3]}
        self.assertEqual(fourth_arm_interval(rows, measured), (q[ARMS[-1]],)*2)
        self.assertEqual(fourth_arm_interval(rows, measured, [F(1,2)]*2), (q[ARMS[-1]],)*2)
        with self.assertRaisesRegex(ValueError,'conflict'):
            fourth_arm_interval(rows, dict(zip(ARMS[:3], [(1,1),(0,0),(0,0)])))

    def test_common_finder_different_wait_or_receipt_state(self):
        a,b = ARMS[:2]
        rows = known(ARMS, [[1,2],[4,5],[9]], {a:'early/executor-near', b:'late/executor-far'})
        self.assertEqual(contrast_bounds([rows],{a:1,b:-1}),(-1,1))
        rows = known(ARMS, [[1,2],[4,5],[9]], {})
        self.assertEqual(len(set(rows.values())),4)

    def test_tied_or_overlapping_windows_do_not_prove_a_winner(self):
        for window in [Window(1,1),Window(1,3),Window(1)]:
            rows = event_rows([[window],[Window(1,2)]], {(0,0):'same'}, horizon=10,hold=2)
            self.assertEqual(rows[(0,0)][0],'private')
        rows = event_rows([[Window(2)],[Window(8)]], {(0,0):'same'},horizon=8,hold=1)
        self.assertEqual(rows[(0,0)][0],'private')

    def test_last_viable_registration_includes_first_receipt_hold_interval(self):
        rows = event_rows([[Window(5,5),Window(6),Window(8)]], {}, horizon=8,hold=3)
        self.assertEqual(contrast_bounds([rows],{(0,):1}),(0,1))
        self.assertIsNone(rows[(1,)])
        self.assertIsNone(rows[(2,)])
        self.assertEqual(contrast_bounds([rows],{(1,):1,(2,):-1}),(0,0))

    def test_mixture_weights_are_common_and_unknown_weights_use_extremes(self):
        a,b = ARMS[:2]
        r1,r2 = {a:'x',b:None}, {a:'x',b:'x'}
        self.assertEqual(contrast_bounds([r1,r2],{a:1,b:-1},[F(1,3),F(2,3)]),(0,F(1,3)))
        self.assertEqual(contrast_bounds([r1,r2],{a:1,b:-1}),(0,1))
        with self.assertRaises(ValueError):
            contrast_bounds([r1,r2],{a:1},[1,1])

    def test_standby_proxy_opposite_rankings_and_distance_ties(self):
        near, far = (1.5,1.5,1.5), (10.5,1.5,1.5)
        unused = (19.5,19.5,7.5)
        menus = [[near,far],[near,far],[unused]]
        c = standby_contrasts(menus, near, [.1]*3)
        self.assertEqual((c['0']['min_prefix'], c['0']['max_prefix']), (0,1))
        self.assertEqual((c['1']['min_prefix'], c['1']['max_prefix']), (-1,0))
        c = standby_contrasts([menus[0],[unused,unused],[unused]],(6,1.5,1.5),[.1]*3)
        self.assertEqual((c['0']['min_prefix'],c['0']['max_prefix']),(0,0))

    def test_hidden_outcomes_do_not_enter_prediction(self):
        menus = [dict(boat=i, step=0, points=[[1,2,3],[3,4,5]]) for i in range(3)]
        snapshot = dict(prefix=dict(menus=menus),initial=dict(standby=[2,3,4]),
                        protocol=dict(ranges=[2,2,2],bias=.1,horizon=400,hold=5))
        original = study.prediction(dict(snapshot=snapshot))
        snapshot['initial'].update(target='secret', task='secret', kinematics='secret')
        snapshot.update(episode='secret',intervals='secret')
        self.assertEqual(study.prediction(dict(snapshot=snapshot)), original)
        self.assertEqual(original['contrasts']['0']['native_bounds'],[-1,1])

    def test_scoring_keeps_zero_evidence_unresolved(self):
        self.assertEqual(study.assess([0]*4,1)['verdict'],'unresolved')
        self.assertEqual(study.assess([1]*4,1)['verdict'],'unresolved')
        self.assertEqual(study.assess([-1]*80,1)['verdict'],'contradicted')
        self.assertEqual(study.assess([1]*80,1)['verdict'],'supported')
        with self.assertRaises(ValueError):
            study.assess([],1)
