# Path-option compute reuse, 2026-11-12

This revision changes compute scheduling in the existing path-option algorithm.
Network modules, parameter names/shapes/initialization, rewards, options, replay,
batch size, update schedule, optimizers, targets, masks and sampling are retained.
There is no agent batching, route truncation, candidate reduction, cross-step
cache or environment change. Execution remains CPU and preserves sampling order.

The optimized learner reuses policy encodings in no-grad inference/target queries,
target critic encodings for two event-region queries before any optimizer step,
and post-critic detached values for the actor objective. It recomputes after the
critic optimizer step. Actor policy graphs remain separate: an initial attempt
to merge them changed floating-point gradient accumulation; Adam amplified a
near-zero action-bias gradient. That candidate was discarded rather than relaxing
the comparison tolerance. Encoder calls are 4 -> 3 per distribution decision and
16 -> 9 per full update. Grad-enabled distribution queries retain both actor paths.

The independent reference is the frozen v3 networks/learner source, hash-pinned in
checks/test_path_option_compute.py, used only in synthetic verification tools.
The checks compare all eight variants, six interventions, train/eval network modes,
online/target outputs, masks/forced actions, categorical decisions and RNG order.
Four sequential updates per temporary model compare losses, every gradient before
and after clipping, model/target weights, Adam state and RNG. Float comparisons
use rtol=1e-5, atol=1e-7; actual maximum differences are recorded, not assumed zero.
Native environment construction is forbidden during the verification driver.

Reproduce from the Active project root, using a fresh output directory:

```powershell
python -B -m tools.benchmark_path_option_compute --output work/collision_terminal/path_option_compute_benchmark_fresh --repeats 3 --execute
python -B -m tools.validate_path_option --output work/collision_terminal/path_option_compute_validation_fresh --execute
```

Timing compares frozen and optimized code in one process, CPU single-thread,
alternating pair order, batch=32 and 16/64 vertices per each of eight paths.
Each temporary model has at most four complete updates; timing warmup is forward
only. The benchmark measures network work, excluding environment, planning, JSON
logging and development evaluation. It is neither native throughput nor a server
speed guarantee. The source manifests v1-v3 and historical evidence stay immutable;
v4 plus a precise evolution record bind this revision. Old checkpoint/evidence
identities are not upgraded. Linux native smoke under the current v4 source is
still required for research admission, followed by explicitly requested training.

Final results and the created-file ledger are stored under
verification/path_option_compute_v1/20261112_v1/. No long training, environment
rollout, checkpoint restore, commit or push is part of this task.
