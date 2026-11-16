from dataclasses import dataclass
import numpy as np
from chapter3_bser.r123.lifecycle import Lifecycle as ParentLifecycle, ProgressManager as ParentManager


@dataclass
class Attempt:
    endpoint: np.ndarray
    started: int
    progress_step: int
    position: np.ndarray
    tracking: np.ndarray


class Lifecycle(ParentLifecycle):
    def __init__(self, control):
        super().__init__(control)
        self.attempts={}

    def accept(self, agent, point, phase, step, position):
        self.attempts.pop(agent,None)
        super().accept(agent,point,phase,step,position)

    def enter_recovery(self, agent, step, endpoint):
        task=self.tasks[agent]
        prior=self.attempts.get(agent)
        if prior is not None and np.linalg.norm(prior.endpoint-endpoint)<.5:
            return  # resubmitting one attempt buys neither time nor another budget
        if task.released or task.phase==3 or task.recoveries>=self.control['max_recoveries']:
            raise ValueError('cannot begin recovery for inactive/exhausted task')
        super().enter_recovery(agent,step,endpoint)
        self.attempts[agent]=Attempt(np.asarray(endpoint).copy(),step,step,
                                    task.last_position.copy(),np.asarray(endpoint).copy())

    def attempt_progress(self, agent, position, tracking, step):
        a=self.attempts[agent];p=np.asarray(position)
        delta=a.tracking-a.position
        if np.linalg.norm(delta)>1e-6 and np.dot(p-a.position,delta/np.linalg.norm(delta))>=self.control['progress_distance']:
            a.progress_step=step
        a.position=p.copy();a.tracking=np.asarray(tracking).copy()
        return a.progress_step


class ProgressManager(ParentManager):
    def status(self, agent, position, velocity, phase, step, endpoint_free, target_velocity=None):
        if phase==3:return 'FROZEN'
        reason=super().status(agent,position,velocity,phase,step,endpoint_free,target_velocity)
        c=self.current.get(agent);a=self.lifecycle.attempts.get(agent)
        if reason in ('CONTINUE','STALLED') and c is not None and c.mode=='RECOVER' and a is not None:
            progress=self.lifecycle.attempt_progress(agent,position,c.route.tracking(position),step)
            return 'STALLED' if step-progress>=self.options['stagnation_steps'] else 'CONTINUE'
        return reason
