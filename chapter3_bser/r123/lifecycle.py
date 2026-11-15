"""Task ownership and bounded recovery, independent of learned termination."""
from dataclasses import dataclass
import numpy as np
from chapter3_bser.path_option.path_context import GoalManager


def arrived(position, velocity, endpoint, phase, pose_free, clearance, target_velocity=(0.,0.,0.)):
    if phase==3 or not pose_free or clearance<=.52 or np.linalg.norm(np.asarray(position)-endpoint)>.75:
        return False
    return phase==0 or np.linalg.norm(np.asarray(velocity)-(target_velocity if phase==2 else 0.))<=.2


@dataclass
class Task:
    identity: int
    point: np.ndarray
    phase: int
    created: int
    progress_step: int
    last_position: np.ndarray
    last_tracking: np.ndarray
    recoveries: int=0
    released: bool=False
    release_reason: str=''
    recovery_start: int | None=None
    root_only: bool=False


class Lifecycle:
    def __init__(self, control):
        self.control=control; self.tasks={}; self.serial=0; self.events=[]

    def task_id(self, agent):
        return self.tasks[agent].identity if agent in self.tasks else 0

    def accept(self, agent, point, phase, step, position):
        self.serial+=1
        self.tasks[agent]=Task(self.serial,np.asarray(point).copy(),phase,step,step,
                              np.asarray(position).copy(),np.asarray(point).copy())
        self.events.append(dict(agent=agent,step=step,event='TASK_ACCEPT',task_id=self.serial,phase=phase))

    def progress(self, agent, position, tracking, step):
        task=self.tasks[agent];p=np.asarray(position)
        direction=task.last_tracking-task.last_position
        if np.linalg.norm(direction)>1e-6:
            forward=float(np.dot(p-task.last_position,direction/np.linalg.norm(direction)))
            if forward>=self.control['progress_distance']:
                task.progress_step=step
        task.last_position=p.copy();task.last_tracking=np.asarray(tracking).copy()

    def enter_recovery(self, agent, step, endpoint):
        task=self.tasks[agent];task.recoveries+=1
        if task.recovery_start is None:task.recovery_start=step
        self.events.append(dict(agent=agent,step=step,event='RECOVERY_ENTER',task_id=task.identity,
                                attempt=task.recoveries,endpoint=np.asarray(endpoint).tolist()))

    def release(self, agent, step, reason):
        task=self.tasks[agent]
        if not task.released:
            self.events.append(dict(agent=agent,step=step,event='TASK_RELEASE',task_id=task.identity,reason=reason))
            task.release_reason=reason
        task.released=True


class ProgressManager(GoalManager):
    """Shared arrival correctness; R2 uses progress across goal resubmissions."""
    def __init__(self, options, lifecycle, r2):
        super().__init__(options); self.lifecycle=lifecycle;self.r2=r2;self.public_pose={}

    def status(self, agent, position, velocity, phase, step, endpoint_free, target_velocity=None):
        c=self.current.get(agent)
        if c is None:return 'INITIAL'
        if phase!=c.phase:return 'PHASE_CHANGED'
        if not endpoint_free:return 'KNOWN_INVALID'
        pose_free,clearance=self.public_pose.get(agent,(False,0.))
        reached=arrived(position,velocity,c.endpoint,phase,pose_free,clearance,
                        (0.,0.,0.) if target_velocity is None else target_velocity)
        if reached:return 'REGION_ARRIVED' if phase==0 else 'HOLD'
        if phase in (1,2) and pose_free and clearance>.52 and np.linalg.norm(position-c.endpoint)<=.75:
            return 'SETTLING'
        if not c.route.reachable:return 'NO_ROUTE'
        if self.r2:
            self.lifecycle.progress(agent,position,c.route.tracking(position),step)
            stalled=step-self.lifecycle.tasks[agent].progress_step>=self.options['stagnation_steps']
            return 'STALLED' if stalled else 'CONTINUE'
        distance=float(np.linalg.norm(position-c.endpoint))
        if distance<c.best_distance-self.options['progress_distance']:
            c.best_distance,c.progress_step=distance,step
        return 'STALLED' if step-c.progress_step>=self.options['stagnation_steps'] else 'CONTINUE'
