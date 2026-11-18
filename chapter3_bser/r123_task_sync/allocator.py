"""D2 uses accepted execution for fixed agents, preserving separate proposals."""
from dataclasses import asdict,replace
from time import perf_counter
from chapter3_bser.d2.allocator import EffectAllocator
from chapter3_bser.online.types import SearchAssignment
from chapter3_bser.r123_task_sync.receipts import snapshot


class SyncAllocator(EffectAllocator):
    def receipts(self,state):return snapshot(self.owner,state)

    def record(self,state,scope,receipts,used):
        rows=[]
        for i,c in used.items():
            r=receipts.get(i)
            if r is None:continue
            expected=r.candidate()
            consistent=c.waypoint==expected.waypoint and c.path_points.tolist()==expected.path_points.tolist()
            if not consistent:raise RuntimeError('execution receipt/scoring mismatch')
            rows.append(dict(agent=i,task_id=r.task_id,status=r.status,endpoint=list(c.waypoint),
                route_id=r.route_id,route_index=r.route_index,consistent=consistent))
        self.owner.sync_scoring.append(dict(step=state.step,scope=scope,used=rows))

    def _proposal(self,state,candidates,frozen,standbys,executor,trigger):
        if not self.owner.task_sync_enabled or state.target_found:
            return super()._proposal(state,candidates,frozen,standbys,executor,trigger)
        start=perf_counter();receipts=self.receipts(state)
        current=self.owner.controller.inner.current_allocation
        originals={v.agent_id:v for v in current.search_assignments}
        # A live commitment is fixed during this solve. Suggestions cannot
        # silently replace it before the existing lifecycle accepts them.
        active={i for i,r in receipts.items() if r.status in ('ACTIVE','UNREACHABLE') and i in originals}
        fixed={c.agent_id:c for c in frozen}
        for i in set(fixed)|active:
            if i in receipts:fixed[i]=receipts[i].candidate()
        choices=tuple(c for c in candidates if c.agent_id not in fixed)
        self.sync_fixed_context=(state.step,frozenset(fixed))
        self.record(state,'proposal_fixed',receipts,fixed)
        proposed=super()._proposal(state,choices,tuple(fixed[i] for i in sorted(fixed)),standbys,executor,trigger)
        # Only scoring uses physical connector/hold receipts. Never install a
        # connector, position placeholder or receipt ID as a new root proposal.
        restored=tuple(originals.get(c.agent_id,c) if c.agent_id in fixed else c for c in proposed.search_assignments)
        self.owner.sync_seconds+=perf_counter()-start
        return replace(proposed,search_assignments=restored)

    def rescore(self,state,allocation,*,finalize=False):
        if not self.owner.task_sync_enabled or state.target_found or allocation.search_frozen:
            return super().rescore(state,allocation,finalize=finalize)
        start=perf_counter();receipts=self.receipts(state);used={};effective=[]
        for item in allocation.search_assignments:
            r=receipts.get(item.agent_id)
            # A newly proposed task for a released agent is prospective in
            # proposal finalization only. Current-plan rescore always uses
            # factual execution (including stationary sensing while waiting).
            fixed_step,fixed_ids=getattr(self,'sync_fixed_context',(-1,frozenset()))
            if r is not None and (not finalize or r.status in ('ACTIVE','UNREACHABLE') or (fixed_step==state.step and item.agent_id in fixed_ids)):
                c=r.candidate();used[item.agent_id]=c
                effective.append(SearchAssignment(c.agent_id,c.candidate_id,c.waypoint,
                    tuple(map(tuple,c.path_points)),c.physical_travel_time,(),c.planning_cost))
            else:effective.append(item)
        self.record(state,'rescore_final' if finalize else 'rescore_current',receipts,used)
        scored=super().rescore(state,replace(allocation,search_assignments=tuple(effective)),finalize=finalize)
        self.owner.sync_seconds+=perf_counter()-start
        return replace(scored,search_assignments=allocation.search_assignments)
