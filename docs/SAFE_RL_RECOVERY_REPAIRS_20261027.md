# Safety recovery repairs1/2/5 — 2026-10-27

Parent:5ed20460e5b6a98642a39b6844790e82d38f3bf1. Current reviewed source:088a5dc07f206d3855aa548c5da0d64aca5d8317efdfdd5dffcf2cc1329c0c41.

The user authorized only audit findings1/2/5. Original D2, B2/B3 repair-v2,
4 agents,28D/3D/124D task contract,gamma=.95,rewards,dynamics,target and
termination semantics remain protected. New recovery schema:auv.invalid_goal_safe_rl.v2.
Every parent manifest,evidence file and checkpoint identity is preserved.

##1: Exit an already violated margin

Ordinary public voxel/hit clearance checks stay strict. A separate BUFFER_EXIT
candidate may override the zero isotropic speed cap only when its first action
moves outward from all initially offending public supporting planes, under the
estimated flow and four constant XY flow corners±.18. The finite backup follows
an outward .35m/s request and then brakes, within32 model steps. First-step
predicted speeds must remain at most .5m/s horizontally and .4m/s vertically;
commands stay within original acceleration limits. Existing high inward speed
or conflicting evidence can make egress infeasible and is never labeled checked.

Voxel membership uses the same axis-aligned inflated geometry as ordinary
checks. Egress overlaps the nominal .52m margin by .10m and aims past .67m
before braking, avoiding the corner-classification and partial-exit deadlocks.
Initially enclosing coarse occupied voxels are ambiguous: egress requires near
scan evidence and logs that ambiguity. No truth obstacle is queried by control.
Constant-flow probes and supporting planes do not certify unseen surfaces,
arbitrary time-varying disturbance or a terminal invariant.

##2: Release staging overrides

A stage reaching the .25m tracker threshold, or expiring after40 steps,
releases its old route and schedules a fresh query from the current continuous
pose. Query interval5 is retained. Retry intent survives throttling; the original
semantic goal is retried when public conflict clears. Alternative stages cannot
be the already reached nearby point. Goal or phase changes discard stale
routes. Failed repair holds a latched anchor and retries, without claiming task
completion. An actually impossible task can still time out.

##5: Collision and progress acceptance

The bounded checker rejects collisions, incomplete per-role traces, nonfinite
or physically illegal controls, mislabeled uncertified actions and stalled
timeouts in the curated recoverable safety cases. The new independent fixture
places all four roles1.5mm outside obstacle faces, supplies each a bad goal in
its own obstacle, and retains the original sensor/flow/dynamics. Truth geometry
only constructs the offline fixture and supports posthoc checks. Every role must
move away on the first step, exercise checked BUFFER_EXIT and leave the original
.52m margin within32 steps. This replaces the untrained-policy native fixture.

Current Windows verification:363 regression tests;7 initializations,
224 physical steps;zero native optimizer updates and
zero checkpoint-file restores. Same-host baseline repeat is exact; external
nominal composition retains numerical parity. Four curated safety cases passed
collision and motion checks. Near-buffer exit counts:{"agent_0:BUFFER_EXIT": 10, "agent_1:BUFFER_EXIT": 10, "agent_2:BUFFER_EXIT": 10, "agent_3:BUFFER_EXIT": 10}.
These are32-step mechanism results, not full82/96-step reproduction,400-step
acceptance or demonstrated task success. Linux must validate this exact package
before explicit manual OLD0002/0006 evaluation.

## Deferred learning findings

Findings3(direction-aliasing input),4(independent horizon-head selector) and
6(positive replay eviction) remain unresolved. Feature AST,learner.py and
dataset.py are independently protected against changes. New safety train and
learned-policy load/evaluation are suspended before admission or environment
creation. Rule evaluation and explicit nominal-data collection remain available;
the original B2/B3 entrypoints are preserved. No RL novelty or performance gain
is established by this repair.

Evidence:verification/safe_rl_recovery_v2/088a5dc07f206d3855aa548c5da0d64aca5d8317efdfdd5dffcf2cc1329c0c41/.
Linux commands are supplied directly in the Codex reply, not in the README.
