import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools.work1_linux_dispatch_maps import choose, decide_branch, earliest_step, metrics
from tools.work1_linux_dispatch_report import observe, paired_metrics, summarize, window


class DispatchChecks(unittest.TestCase):
    def test_zero_evaluable_families_report_preserves_denominators(self):
        import gzip
        with TemporaryDirectory() as temp:
            out=Path(temp)
            for i in range(4):
                folder=out/f'family_{i+1:02d}'/'spatial'
                folder.mkdir(parents=True)
                d=dict(plans=[[]],offers={},eligibility={},belief=[1.],gate='incomplete_team',
                       choices={},branch_reason='incomplete_team',branch=None)
                (folder/'prediction.json').write_text(json.dumps(d),encoding='utf8')
                (folder/'complete.json').write_text('{"steps":1}',encoding='utf8')
                with gzip.open(folder/'physics.jsonl.gz','wt',encoding='utf8') as f:
                    f.write(json.dumps(dict(step=1,commands=[None]*4,checked=False,hits=[[],[],[]],
                               found=False,found_before=False,reason='collision',terminal=True)))
            summarize(out)
            result=json.loads((out/'results.json').read_text(encoding='utf8'))
            self.assertEqual(result['complete_pairs'],0)
            self.assertEqual(result['gate_counts'],{'incomplete_team':4})
            self.assertEqual(result['direction']['history']['scorable'],0)
            self.assertTrue((out/'REPORT.md').is_file())

    def test_time_bound_is_only_necessary(self):
        spec=dict(v_xy_max=3.,v_z_max=4.)
        self.assertEqual(earliest_step([0,0,0],[17,0,0],1,spec,.2),16)
        self.assertEqual(earliest_step([0,0,0],[17.01,0,0],1,spec,.2),17)
        self.assertEqual(earliest_step([0,0,0],[0,0,0],1,spec,.2),1)

    def test_shared_scenario_union_and_overlap(self):
        first=np.array([[True,False],[False,True]])
        empty=np.zeros_like(first)
        w=np.array([.5,.5])
        keys=['a','b','c']
        offers={k:{'cell':0} for k in keys}
        draws={k:{'history':(v,v,w)} for k,v in zip(keys,(first,first,empty))}
        s=metrics(keys,offers,draws,np.array([.6,.4]),'history')
        np.testing.assert_allclose(s['boats'][0],[.5,.5,0])
        np.testing.assert_allclose(s['coverage'],[.5,.5])
        np.testing.assert_allclose(s['region'],[1,1])

    def test_ties_and_unknowns_preserve_spatial(self):
        def pick(intervals):
            return choose([{'coverage':i} for i in intervals],'coverage')
        self.assertEqual(pick([[.1,.1],[.1,.1]]),0)
        self.assertEqual(pick([[.1,.3],[.2,.2]]),0)
        self.assertEqual(pick([[.1,.1],[.2,.2]]),1)

    def test_diagnostic_swap_is_not_a_policy_switch(self):
        plans=[('a','b'),('c','d')]
        offers={k:dict(cell=g,boat=i,support='route_available',lower_step=8)
                for k,i,g in [('a',0,1),('b',1,2),('c',0,2),('d',1,1)]}
        choices={m:{'region':0,'coverage':0} for m in ('history','bins')}
        self.assertEqual(decide_branch(plans,offers,choices,0),(1,'competition_diagnostic'))
        choices['bins']['coverage']=1
        self.assertEqual(decide_branch(plans,offers,choices,0),(1,'competition_policy'))

    def test_discovery_censoring_and_own_command_clock(self):
        goals=[[0,0,0],[1,0,0],[2,0,0]]
        expected=[[str(i),g] for i,g in enumerate(goals)]
        d=dict(plans=[['0','1','2']],belief=[.2,.3,.5],
               offers={str(i):dict(boat=i,cell=i,goal=g) for i,g in enumerate(goals)})
        row=dict(step=1,commands=expected,hits=[[0],[0],[]],receipts=[{'accepted':True}]*3,
                 checked=True,found=True,terminal=False,reason='running')
        actual=observe([row],d,0,16)
        self.assertIs(actual['boats'][0]['hit'],True)
        self.assertIsNone(actual['boats'][1]['hit'])
        self.assertFalse(actual['complete'])
        self.assertEqual(actual['stop'],'discovery')
        first=dict(row,found=False)
        second=dict(first,step=2,commands=[None,*expected[1:]])
        self.assertEqual(window([first,second],[0,1,2],expected,2)[1],'new_or_unissued_command')
        self.assertEqual(window([first,second],[1],expected,2)[1],'complete')

    def test_competition_requires_accepted_observations_on_both_boats(self):
        def side(i,accepted,hit):
            return dict(complete=False,boats=[dict(boat=i,cell=9,accepted=accepted,hit=hit,first=1 if hit else None)])
        d=dict(scores={m:[None,None] for m in ('history','bins')},eligibility={'9':{}})
        for accepted,hit,outcome in [(True, True,'both_swept'),(False,True,'admission_unknown'),
                                     (True,None,'censored'),(True,False,'at_least_one_observed_miss')]:
            r=paired_metrics(d,1,side(0,True,True),side(1,accepted,hit))
            self.assertIsNone(r['observed_delta'])
            self.assertEqual(r['exchanges'][0]['status'],outcome)
