"""Optional read-only failure observer; returns the original safety verdict."""
import numpy as np
from chapter3_bser.reach_avoid.scan_geometry import segment_pieces


def explain_segment(g,p,q,extra):
    p,q=np.asarray(p,float),np.asarray(q,float)
    result=dict(start=p.tolist(),end=q.tolist(),extra=float(extra),padding=float(g.padding),
        public_geometry_only=True,ground_truth_geometry_used=False)
    if not np.isfinite([p,q]).all():return dict(result,cause='nonfinite')
    if np.any(np.minimum(p,q)<g.raw.lower) or np.any(np.maximum(p,q)>g.raw.upper):return dict(result,cause='bounds')
    starts,ends,keys=segment_pieces(p,q)
    blocked=[tuple(k) for k in keys if tuple(k) in g.subcells.blocked]
    unknown=np.array([tuple(k) not in g.subcells.free for k in keys])
    result.update(blocked_subcells=[list(map(int,k)) for k in sorted(set(blocked))],
        unknown_piece_count=int(unknown.sum()),total_pieces=len(keys))
    observed=g.observed_path(p,q,extra)
    if not observed:
        # Do not invent a precise occupied box if the original path was vetoed
        # by its coarse public geometry; this records the actual query branch.
        return dict(result,cause='blocked_subcell' if blocked else 'unobserved_coarse_geometry')
    if len(g.points):
        d=q-p;t=np.clip((g.points-p)@d/max(float(d@d),1e-30),0,1)
        gap=np.linalg.norm(g.points-(p+t[:,None]*d),axis=1)
        bad=np.flatnonzero(gap<=g.padding+extra)
        return dict(result,cause='scan_hit_padding' if len(bad) else 'unclassified_geometry_veto',
            nearest_hit_distance=float(gap.min()),offending_hit_count=len(bad),offending_hits=g.points[bad[:8]].tolist())
    return dict(result,cause='unclassified_geometry_veto')


class ProbeGeometry:
    def __init__(self,geometry):self.geometry=geometry;self.failure=None
    def __getattr__(self,name):return getattr(self.geometry,name)
    def segment(self,p,q,extra=0.):
        verdict=self.geometry.segment(p,q,extra=extra)
        if not verdict:
            try:self.failure=explain_segment(self.geometry,p,q,extra)
            except Exception as exc:self.failure=dict(cause='REVIEW_REQUIRED',diagnostic_error=type(exc).__name__)
        return verdict


class FailureObserver:
    """At most two recent steps in memory; emit details only for interventions."""
    def __init__(self,runtime):
        self.runtime=runtime;self.original=runtime.guard.check;self.pending={};self.records=[];self.calls=0
        runtime.guard.check=self.check

    def check(self,p,v,u,spec,geometry,flow):
        proxy=ProbeGeometry(geometry)
        answer=self.original(p,v,u,spec,proxy,flow)
        self.calls+=1
        step=self.runtime.step
        for key in list(self.pending):
            if key[0]<step-1:del self.pending[key]
        if not answer[0]:
            key=(step,float(spec['a_xy_max']))
            self.pending.setdefault(key,[]).append(dict(command=np.asarray(u).tolist(),flow=np.asarray(flow).tolist(),
                reason=dict(answer[1]),failure=proxy.failure))
        return answer

    def collect(self,trace):
        from chapter3_bser.reach_avoid.options import SPECS
        for row in trace:
            if row['phase']!=3 and (row['emergency'] or row.get('navigation_blocked',False)):
                self.records.append(dict(step=row['step'],agent=row['agent'],phase=row['phase'],
                    emergency=row['emergency'],navigation_blocked=row.get('navigation_blocked',False),
                    chosen_action=row['decision']['action'],position=row['position'],velocity=row['velocity'],
                    action_diagnostics=row['action_diagnostics'],
                    rejected_checks=self.pending.get((row['step'],float(SPECS[row['agent']]['a_xy_max'])),[])))

    def close(self):self.runtime.guard.check=self.original


from chapter3_bser.r123_task_sync.runtime import RuleRuntime as ParentRuntime


class ObservedRuntime(ParentRuntime):
    """Original task-sync control, plus separate failure records after each step."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.failure_observer=FailureObserver(self)
    def advance(self,*args,**kwargs):
        result=super().advance(*args,**kwargs)
        self.failure_observer.collect(self.trace[-4:])
        return result
    def summary(self):
        result=super().summary()
        result['failure_diagnostics']=dict(check_calls=self.failure_observer.calls,records=self.failure_observer.records,
            changes_control=False,public_geometry_only=True)
        return result
    def close(self):
        if hasattr(self,'failure_observer'):self.failure_observer.close()
        return super().close()
