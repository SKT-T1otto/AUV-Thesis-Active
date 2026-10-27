"""Public candidate proposal; learned values, not this proposal, select goals."""
import numpy as np
from chapter3_bser.reach_avoid.goals import Goal, SEARCH, STANDBY, CAPTURE, VIA, cell_status, goal_reached
from chapter3_bser.reach_avoid.features import DIRECTIONS


def candidate_goals(packet, state, semantic, geometry, options):
    all_goals, all_masks = [], []
    for i in range(4):
        p = packet.positions[i]
        phase = packet.phases[i]
        kind = STANDBY if phase == 1 else SEARCH if phase == 0 else VIA
        original = Goal(tuple(semantic[i]), kind=kind)

        def admissible(goal):
            free, occupied, inside = cell_status(packet, goal.ref)
            return bool(inside and free and not occupied
                and np.linalg.norm(np.asarray(goal.ref) - p) <= 4.5
                and geometry.distance(goal.ref) > geometry.padding + .05
                and geometry.segment(p, goal.ref))

        goals, mask = [original], [admissible(original)]
        if phase == 3:
            goals = [Goal(tuple(p), kind=VIA)]
            mask = [admissible(goals[0])]
        elif phase == 1:
            stay = Goal(tuple(p), kind=STANDBY)
            goals.append(stay);mask.append(admissible(stay))
        elif phase == 2:
            # Capture is a REGION about the public target estimate. Its center
            # need not itself be a safe AUV point. Safe reference offsets are
            # alternative approaches to the SAME moving goal set.
            center = packet.intents[i]
            refs = [center, *[center + d * .65 for d in DIRECTIONS]]
            refs.sort(key=lambda x: float(np.linalg.norm(x - p)))
            for ref in refs:
                goal = Goal(tuple(center), tuple(packet.target_velocity), CAPTURE, tuple(ref))
                if admissible(goal):
                    goals.append(goal);mask.append(True)
                if len(goals) >= 3:
                    break
        centers = np.asarray(state.grid.cell_centers)
        distance = np.linalg.norm(centers - p, axis=1)
        score = np.linalg.norm(centers - packet.intents[i], axis=1) + .15 * distance
        for index in np.argsort(score):
            if not options['goal_radius'] + .1 < distance[index] <= 4.5:
                continue
            goal = Goal(tuple(centers[index]), kind=kind if phase in (0, 1) else VIA)
            if admissible(goal) and all(np.linalg.norm(np.asarray(g.ref) - goal.ref) > 1e-5 for g in goals):
                goals.append(goal);mask.append(True)
            if len(goals) == options['candidates']:
                break
        goals, mask = goals[:options['candidates']], mask[:options['candidates']]
        for j,g in enumerate(goals):
            # A completed moving option is not an attractive "100% reachable"
            # replacement for doing work. Standby/capture terminal regions have
            # their own hold policy and are intentionally retained.
            if g.kind in (SEARCH,VIA) and phase!=3 and goal_reached(packet,i,g,options):mask[j]=False
        if phase==1 and len(goals)>1:
            current_cost=np.linalg.norm(p-packet.intents[i])
            better=any(mask[j] and np.linalg.norm(np.asarray(g.ref)-packet.intents[i])<current_cost-options['goal_radius'] for j,g in enumerate(goals) if j!=1)
            if better:mask[1]=False
        while len(goals) < options['candidates']:
            goals.append(original);mask.append(False)
        all_goals.append(np.array([g.array() for g in goals]))
        all_masks.append(mask)
    return np.asarray(all_goals, np.float32), np.asarray(all_masks, bool)


def rule_goal(packet, agent):
    valid = np.flatnonzero(packet.goal_mask[agent])
    if not len(valid):
        return 0
    goals = [Goal.from_array(packet.goals[agent, j]) for j in valid]
    scores = [float(np.linalg.norm(np.asarray(g.ref) - packet.intents[agent]))
              + .1 * float(np.linalg.norm(np.asarray(g.ref) - packet.positions[agent])) for g in goals]
    return int(valid[int(np.argmin(scores))])
