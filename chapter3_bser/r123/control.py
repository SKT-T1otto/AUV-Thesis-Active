"""Goal-conditioned action domains. No free probes during ordinary R1 tracking."""
import numpy as np
from chapter3_bser.path_option.runtime import make_bank as old_bank
from chapter3_bser.reach_avoid.goals import Goal, base_action
from chapter3_bser.reach_avoid.options import SPECS, CONTRACT
from chapter3_bser.reach_avoid.scan_geometry import refined_buffer_exit
from chapter3_bser.safe_rl.dynamics import brake, limits, integrate
from chapter3_bser.r123.options import CONTROL


def conditional_banks(packet, agent, tracks, geometry, guard, r1):
    if not r1:
        a, m, f, e = old_bank(packet, agent, tracks, geometry, guard)
        return (np.repeat(a[None], 8, axis=0), np.repeat(m[None], 8, axis=0),
                np.full(8, f), np.full(8, e))
    spec = SPECS[agent]; p = packet.positions[agent]; v = packet.velocities[agent]
    scale = limits(spec); flow = packet.flow[agent]
    b = brake(v, spec, CONTRACT['dt'], flow)/scale
    exit_action = refined_buffer_exit(p, v, spec, geometry, guard.contract, guard.options, flow)
    caps = guard._caps(p, v, spec, geometry)
    violation = max(0., np.linalg.norm(v[:2])-caps[0])+max(0., abs(v[2])-caps[1])
    banks=[]; masks=[]; forced=[]; emergencies=[]
    checked={}
    for track in tracks:
        goal = Goal(tuple(track), tuple(packet.target_velocity) if packet.phases[agent] == 2 else (0.,0.,0.))
        base = base_action(packet, agent, goal)
        a = np.zeros((16,3), np.float32); a[0] = base
        k=1
        for axis in range(3):
            for sign in (-1,1):
                a[k] = np.clip(base+np.eye(3)[axis]*sign*CONTROL['residual_bound'], -1, 1); k+=1
        a[7] = base  # fixed shape; duplicate is masked, not a second probability mass
        a[8] = b; a[9] = b if exit_action is None else exit_action[0]/scale
        valid=np.zeros(16,bool)
        for k in (*range(7),8):
            key=a[k].tobytes()
            if key not in checked:
                nv=integrate(p,v,a[k]*scale,spec,CONTRACT['dt'],flow)[1]
                next_violation=max(0.,np.linalg.norm(nv[:2])-caps[0])+max(0.,abs(nv[2])-caps[1])
                checked[key]=(next_violation<=1e-6 or next_violation<violation-1e-8) and guard.check(p,v,a[k]*scale,spec,geometry,flow)[0]
            valid[k]=checked[key]
        valid[9]=exit_action is not None
        # Brake is a safety fallback, never a competing ordinary navigation goal.
        if valid[:7].any():
            valid[8]=False
        emergency=not valid.any()
        if emergency:
            valid[8]=True
        f=9 if exit_action is not None else 8 if emergency else -1
        banks.append(a);masks.append(valid);forced.append(f);emergencies.append(emergency)
    return np.array(banks),np.array(masks),np.array(forced),np.array(emergencies)
