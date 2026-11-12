# Path-option compute optimization verification

Current source: `e1477269387a1926d5de250e928cd7914db7affd02fc929edb8416005d5866ef`. Frozen reference: v3 `cf81d42a92a46377d520f5e244a0cb424bc81cc63ead9fc819014c403db552ff`.

## Changes

- Reuse no-grad policy history/path encoding and action/goal scores.
- Reuse target critic encodings for the second event-region query before updates.
- Reuse detached values from the post-critic actor query; never cache across optimizer steps.
- Keep both original actor gradient graphs. A discarded merged-graph candidate failed the strict Adam-updated action-bias comparison; tolerances were not loosened.
- Encoder calls: decision 4 to 3; complete update 16 to 9. No four-agent batching or changes to route queries.

## Equivalence and scope

41 related tests passed (37 prior contracts plus 4 independent compute comparison tests). Comparisons cover all 8 variants, all 6 interventions, train/eval network modes, online/target outputs, forced masks, sampled categorical decisions, and 4 sequential updates per temporary learner. Every raw/clipped gradient, loss, online/target parameter, Adam slot/counter and RNG state is compared. No model is updated more than 4 times.

Maximum absolute errors: `{"benchmark_losses": 0.0, "benchmark_state": 0.0, "global_rng": 0.0, "gradients": 0.0, "initial_state": 0.0, "losses": 0.0, "outputs": 0.0, "sampled_decisions": 0.0, "updated_state": 0.0}`.

Environment initializations: 0. Physical steps: 0. Checkpoint restores: 0. Research training: none. Scoped diff whitespace check passed; the two pre-existing pure-MARL metadata edits were left untouched (global diff check still reports their existing CRLF whitespace).

## Paired timing

Windows CPU single-thread, Torch 2.11.0+cu126. Three independent model pairs per case, alternating before/after order. Each model has 4 timed updates, and inference warmup performs forward calls only. Batch=32; all eight paths have the listed length. These are synthetic network timings, not server/native end-to-end throughput.

| Variant | Vertices/path | Decision before/after ms | Decision speedup | Update before/after ms | Update speedup | Synthetic network step time reduction |
|---|---:|---:|---:|---:|---:|---:|
| A | 16 | 6.18 / 5.02 | 1.23x | 252.58 / 186.00 | 1.36x | 24.2% |
| B | 16 | 2.99 / 2.47 | 1.21x | 157.95 / 118.69 | 1.33x | 23.1% |
| A | 64 | 7.09 / 5.60 | 1.26x | 457.01 / 334.56 | 1.37x | 25.6% |
| B | 64 | 3.34 / 2.84 | 1.17x | 210.62 / 153.82 | 1.37x | 24.5% |

Synthetic network step = four single-agent distribution queries plus one quarter of a batch update. Environment, planning, safety geometry, serialization and evaluation remain unmeasured. No claim of full-run speedup or learning performance follows.

## Integrity and research admission

All 210 protected baseline core/config/provenance/historical-evidence files remain byte-identical. CH3/CH4/CH5 full-tree hash comparisons passed, including 369/103/318 files respectively. Parent manifests v1-v3 and historical evidence were not rewritten.

Current-source Windows synthetic validation passed; native Linux smoke remains pending. Old-source native receipts/checkpoints are not relabeled. Training settings/rewards and network state-dict structure/initialization are preserved. No commit or push.

Reproduction commands are in docs/PATH_OPTION_COMPUTE_20261112.md. Raw paired samples are in benchmark.json; checks are in equivalence.log and synthetic.log; current validation is validation.json. created_files.txt and changed_files.json record this task files.
