"""Tracking both + independently switchable accepted-task synchronization."""
from dataclasses import asdict
from chapter3_bser.r123_tracking.runtime import RuleRuntime as Parent
from chapter3_bser.r123_task_sync.allocator import SyncAllocator
from chapter3_bser.r123_task_sync.receipts import snapshot
from chapter3_bser.r123_task_sync.controller import SyncSearchController


class RuleRuntime(Parent):
    def __init__(self,*args,task_sync=True,**kwargs):
        if kwargs.get('tracking_arm','both')!='both':raise ValueError('sync comparison requires tracking both')
        self.task_sync_enabled=bool(task_sync);self.sync_receipts=[];self.sync_scoring=[]
        self.sync_seconds=0.;self.sync_route_queries=0
        self.sync_templates={};self.sync_planner_views=[]
        super().__init__(*args,**kwargs)
        old=self.controller.inner.allocator
        replacement=SyncAllocator.__new__(SyncAllocator)
        replacement.__dict__.update(old.__dict__)
        self.controller.inner.allocator=replacement
        wrapper=SyncSearchController.__new__(SyncSearchController)
        wrapper.__dict__.update(self.controller.__dict__)
        self.controller=wrapper

    def observe(self):
        fresh=self._cached is None or self._cached[0]!=self.step
        rows=super().observe()
        if fresh:
            for item in self.controller.inner.current_allocation.search_assignments:
                task=self.lifecycle.tasks.get(item.agent_id)
                token=(str(item.candidate_id),tuple(item.waypoint))
                if task is not None and not task.released and task.phase==0 and token==self._task_tokens.get(item.agent_id):
                    prior=self.sync_templates.get(item.agent_id)
                    if prior is None or prior[0]!=task.identity:self.sync_templates[item.agent_id]=(task.identity,item)
            receipts=snapshot(self,self._public_state)
            self.sync_receipts.append(dict(step=self.step,receipts=[asdict(r) for r in receipts.values()]))
        return rows

    def _query(self,*args,**kwargs):
        self.sync_route_queries+=1
        return super()._query(*args,**kwargs)

    def summary(self):
        result=super().summary()
        result['method']='auv.r123.task_sync_rule.v1'
        result['task_sync']=dict(enabled=self.task_sync_enabled,receipts=self.sync_receipts,
            scoring=self.sync_scoring,scoring_calls=len(self.sync_scoring),scoring_seconds=self.sync_seconds,
            planner_views=self.sync_planner_views,route_queries=self.sync_route_queries,receipt_violations=sum(not c['consistent'] for r in self.sync_scoring for c in r['used']))
        return result
