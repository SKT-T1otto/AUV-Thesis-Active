"""Read-only public execution receipts; no queries, controller steps or RNG."""
from dataclasses import dataclass
import numpy as np
from chapter3_bser.types import SearchCandidate


@dataclass(frozen=True)
class Receipt:
    agent: int
    step: int
    task_id: int
    token: tuple | None
    root: tuple
    endpoint: tuple
    points: tuple
    route_id: str
    route_index: int
    route_revision: int
    reachable: bool
    status: str
    seconds: float

    def candidate(self):
        points=np.asarray(self.points,dtype=np.float64).reshape(-1,3)
        points.setflags(write=False)
        indices=np.empty(0,dtype=np.int64);indices.setflags(write=False)
        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        return SearchCandidate(self.agent,f'execution_{self.agent}_{self.task_id}_{self.route_id}',
            self.endpoint,points,indices,length,length,self.seconds,'accepted_execution:'+self.status)


def snapshot(runtime,state):
    result={}
    for a in state.agents:
        i=a.agent_id
        if i>=3:continue
        task=runtime.lifecycle.tasks.get(i);c=runtime.manager.current.get(i)
        if task is None or c is None:continue  # normal initial D2 allocation
        p=tuple(float(x) for x in a.position)
        cached=getattr(runtime,'_cached',None)
        reason=cached[1][i]['reason'] if cached is not None else ''
        if task.phase==3:status='FROZEN'
        elif c.mode=='CONNECTOR' and task.released and reason!='WAIT_TASK':status='CONNECTOR'
        elif task.released:status='RELEASED'
        elif not c.route.reachable:status='UNREACHABLE'
        elif task.identity==0:status='UNASSIGNED'
        else:status='ACTIVE'
        # Never turn a released root or an unreachable path into future search.
        moving=status in ('ACTIVE','CONNECTOR') and c.route.reachable
        points=tuple(tuple(float(x) for x in v) for v in c.route.points) if moving else (p,)
        if not points:points=(p,)
        endpoint=tuple(float(x) for x in c.endpoint) if moving else p
        result[i]=Receipt(i,state.step,task.identity,runtime._task_tokens.get(i),
            tuple(float(x) for x in task.point),endpoint,points,c.route.route_id,
            c.route.index,c.route.revision,bool(c.route.reachable),status,
            float(c.route.estimated_seconds) if moving else 0.)
    return result
