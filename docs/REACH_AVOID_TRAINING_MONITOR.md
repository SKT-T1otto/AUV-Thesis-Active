# Reach-avoid training observation protocol

This revision adds an observer and explicit Linux experiment launcher to the
e613ecb geometry-v2 algorithm. It does not change the learner, public geometry,
controller, auxiliary losses, rewards, dynamics, gamma, actor/critic shapes,
optimizer settings, update schedule, D2 or B2/B3. The parent source and evidence
are frozen in `provenance/baselines/reach_avoid_geometry_20261028_v2`.

The experiment trains only the latest full reach-avoid method. D2_ZERO remains
the original task policy; the independent goal-conditioned actor and twin critics
are the networks being trained. The b2 environment config does not enable the
historical B2 trainer. A new-source mixed nominal/exploratory factual collection
of 32 train episodes supplies the existing replay admission requirements.
If it does not contain both factual reach and collision examples, admission fails
explicitly; data/weights are never relabeled. There are 256 offline updates,
followed by exactly 500 online train episodes, maximum 400 physical steps each.
The existing per-episode online update schedule is retained.

The launcher exports 32 frozen training scenes (seed81001) and 12 separate
development scenes (seed82001). Development IDs and physical fingerprints must
not overlap any bootstrap or future train scene. Training cycles through the
fixed train set; this is not a claim of 500 independent layouts. The original
learning seed20261027 and every option are recorded explicitly. No final test
set is opened, no rule/B2/B3 research arm is launched, and no training checkpoint
is restored.

Development evaluations occur after offline pretraining (episode0), and after
online episodes50,100,...,500: 11 checkpoints x12 episodes. Exploration is off.
The evaluation policy is cloned from memory. Python, NumPy and Torch global RNG
states are restored, and the original learner's policy, target networks,
optimizers, local RNG and train/eval modes are fingerprinted before/after.
Plotting is also wrapped in RNG preservation. Evaluation steps never enter the
training replay, update count, training step budget or the 500 episode count.
The maximum budgets are12,800 bootstrap,200,000 online train and52,800 development
steps. These are manual research budgets, not automated mechanism checks.

## What the observer reports

Append-only journals record optimizer losses, post-clipping gradient norms,
real optimizer counters, per-episode actor/critic parameter changes, original
task returns, success, actual collision, timeout and progress stalls. All agents
and phases are included; executor Search and other-phase collisions and searcher
collisions are counted separately. An early crash receives400 penalized
completion steps, so short failed episodes do not look faster than success.
The root episode journal retains full traces. Only the small in-memory list
used for final success/collision totals is compacted to avoid retaining500
copies of those traces in RAM.

A stall is50 consecutive active steps below0.05 speed and more than0.8m from the
current tracking reference, or uncertified action steps, excluding an achieved
local goal. Inactive captured roles are excluded. This is a reporting heuristic,
not a change to task termination or reward. Goal switching, task waiting and
legitimate recovery can affect it; raw traces remain available for interpretation.

Competing-event predictions are scored only for the next actually executed
action (h=1 Brier scores). A trajectory that switches goals is not naively used
as a32-step calibration label for a fixed-goal value. Neither these scores nor
twin-critic extrema establish a calibrated probability bound.

## Predeclared empirical screen

The last five development points must have success/collision/timeout/stagnation
rate and uncertified-fraction ranges at most0.05, return range at most5% of
max(1,abs(initial post-pretrain return)), and penalized completion-time range
at most20 steps. Tail mean success must be at least0.8, collision at most0.1,
timeout at most0.2 and stagnation at most0.2. Collision cannot regress from the
initial policy by over0.05. Actual online optimizer updates are required.

The report distinguishes insufficient evidence, ongoing change, ineffective
plateau, safety regression, stable effective behavior with online gain, and
stable good behavior without clear online gain. Online gain compares the tail
against the post-pretrain policy, not against an untrained or rule baseline.
The screen does not stop training early. A live report's candidate result is
provisional; a completed-budget candidate requires every planned train episode
and development point plus matching completion and journal hashes.

Twelve scenes give8.33 percentage-point rate increments. Wilson95% intervals
are shown for binomial rates; these are uncertainty descriptions for the fixed
development cases, not independent checkpoint samples or a generalization
guarantee. A single training seed, a flat loss, or500 episodes cannot prove RL
convergence. Multi-seed and untouched test comparisons remain future research.

## Artifacts and validation limits

Each evaluation snapshot has source/options/ablation/update and hash metadata.
Snapshots contain no optimizer or replay state and are for evaluation only.
The best-policy pointer uses development success, collision, timeout, stagnation,
original return and earlier-episode tie ordering. It never renames older weights.
Raw development episodes, metrics and policy files are hashed; a report verifies
those hashes without restoring a model. Live reports consume only newline-ended
journal prefixes and record their lengths/hashes. Reports use fresh directories.

PNG/SVG figures, CSV, HTML and JSON are generated after each evaluation and at
completion. Matplotlib is an optional rendering dependency: if absent, the
CSV/JSON/HTML report still works. Any other plotting or integrity error remains
visible rather than being silently accepted. No Linux research or500ep training
is represented as having run by Windows checks. This revision's checks use
synthetic packets and fake runtimes, with zero native environment initialization
and zero physical steps. Policy evaluation invariance includes comparing the
next actual optimizer update against an unobserved control learner, within four
complete updates per temporary learner.
