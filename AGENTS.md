# Active rebuilt research repository

The 2026-10-19 user request authorizes completing this independent directory
for manual D2, D2+B2 and D2+B3 thesis research runs, preserving the current
AUV-Thesis repair-v2 algorithm and environment behavior regardless of results.
Only core and chapter3_bser implement runtime behavior. Never import the old
checkout, sibling repositories, old tests, historical trainers or outputs.
The focused checks/ and tools/ verify this rebuild and are not runtime imports.

Preserve 4 agents, 28D/3D/124D, gamma=.95, original rewards and weights,
network sizes, optimizers, update schedule, dynamics, target/obstacle process,
discovery, handoff and termination semantics unless the user later requests
an explicitly identified algorithm change. D2 is a zero-RL-residual reference;
it has no learned weights. B2/B3 use their repair-v2 identities.

Changes require precise reviewed source evolution and independent verification.
Keep parent manifests/evidence immutable; never relabel old checkpoints.
Manual research execution requires current rebuild evidence and an explicit
--execute command. No long training or checkpoint restore as an automatic
verification step. Mechanism checks stay within 8 episodes x 32 steps and
4 complete multi-agent updates per temporary model. Never overwrite outputs.
No commit/push without explicit user instruction. Historical E0 and formal
performance acceptance remain separate; do not claim them through this rebuild.
