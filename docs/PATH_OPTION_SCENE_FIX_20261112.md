# Pilot preparation: full-trajectory feasibility repair

The ce73724 Linux run passed its 41 synthetic checks and 64-step native smoke,
but the pilot process exited in `prepare`, before training or job creation.
The previous checks did not execute the real complete preparation pipeline.

The fixed special layout had a full-height wall and target velocity
(.30, -.12, .03). At physical step 200 its target was near (11, 13.2, 6.2).
Seven target positions had no connector in the required oracle component:
target reflection uses .2 m clearance, while planning requires .4 m. The offline
feasibility check correctly rejected these positions; this was not a dependency
or learner error. No feasibility check is removed, loosened or ignored.

Special layouts now use velocity (hypot(.30, .12), 0, .03), retaining speed norm,
400-step horizon, reflection dynamics, obstacles, roles, initial positions,
fault windows and the same feasibility checks. The target stays at y=18, in
the upper corridor outside the expanded walls of all five layouts. This changes
special evaluation trajectories and is explicitly `path_option_challenge_v2`.
No v1 special results are relabeled. Runtime learning, rewards, optimizers,
training options, train/development/normal generation and seeds remain unchanged.

Checks reproduce the original rejection, validate samples of all five layouts
over their full 401-position trajectories, retain the original speed, and ensure
feasibility failures still propagate. The current validator includes these
offline scene checks before its optional native smoke. A separate real `prepare`
integration run generates all 32 training, 12 development, 100 normal and 100
special scenes, checks every feasibility gate and the split/fingerprint contract.
Its frozen plan is also loaded successfully after generation. No environment,
learner training or checkpoint restore is needed for this preparation test.

Progress messages identify the preparation stage; initial scene construction
may take time before training starts. Source v5 precisely records the tooling
and evaluation-data evolution; v4 and earlier evidence stay immutable. Linux
users must generate new v5 validation evidence and fresh pilot output paths.
Existing failed outputs and logs should be retained. Activate the existing AUV
environment before restarting. Research training is still an explicit user run.
