"""Explicit synthetic public packets for algebra/learning tests, not rollouts."""
import copy
from types import SimpleNamespace as NS
import numpy as np
from checks.test_safe_rl import state,SPECS,CONTRACT
from chapter3_bser.reach_avoid.options import validate
from chapter3_bser.reach_avoid.control import ReachAvoidController
from chapter3_bser.reach_avoid.goals import Goal,SEARCH
from chapter3_bser.reach_avoid.learner import ReachAvoidLearner,Sample


def public_controller(options=None,points=()):
    o=validate() if options is None else options
    contract=dict(CONTRACT,eps=1e-6,target_obstacle_clearance=.2,target_max_reflections_per_step=4)
    c=ReachAvoidController(SPECS,contract,o,32)
    for j,p in enumerate(points):c.guard.evidence.hits[(j,0,0)]=(np.array(p,float),0)
    s=state();semantics={i:(9.,3.,4.) for i in range(4)}
    packet=c.observe(s,NS(executor_knows_target=False),np.zeros((4,28)),semantics,())
    return c,packet,s,contract


def learning_fixture():
    o=dict(validate(),batch_size=4,warmup_samples=4,minimum_updates=4)
    c,p,_,_=public_controller(o)
    learner=ReachAvoidLearner(o)
    goal=Goal(tuple(p.positions[0]+[1.,0.,0.]),kind=SEARCH)
    for k in range(8):
        nxt=copy.deepcopy(p);nxt.step=1
        reached=k%2==0
        nxt.positions[0]=np.array(goal.center) if reached else p.positions[0]
        learner.replay.add(Sample(p,nxt,0,goal,np.zeros(3,np.float32),.1,True,not reached,True,reached,False))
    return learner,o,p
