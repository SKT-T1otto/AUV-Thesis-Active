# Pilot preparation repair, 2026-11-12

The Linux launch at ce73724 exited during special_manifest, before any training.
The original full-height wall and diagonal target trajectory pass target .2 m
clearance but enter the planner .4 m margin. The first rejected position is near
(11, 13.2, 6.2) at step 200; seven trajectory points lack a required oracle connector.

The fix preserves the trajectory generator, speed norm, horizon, obstacles,
initial positions, fault windows and every feasibility check. It changes only
special-case velocity direction to (hypot(.30,.12),0,.03), staying at y=18 in the
upper corridor. Special cases are explicitly revision path_option_challenge_v2.
Train/development/normal generator AST, learning/runtime code, core, rewards and
training configuration remain unchanged. Source v5 preserves v1-v4 records and
requires a new current-source Linux validation; old receipts are not relabeled.

Validation:
- 44 local tests and 44 tests in an independent staged-tree checkout passed.
- Original invalid trajectory rejection is reproduced as a regression.
- Five special layouts are checked over all 401 trajectory positions.
- Feasibility exceptions still propagate; there is no retry/skip workaround.
- Real pilot prepare generated 32 training, 12 development, 100 normal and 100
  special scenes, including 20 per each of five categories; frozen plan verified.
- Full preparation took 653.64 seconds locally. Progress now identifies each stage.
- No mission environment initialized, physical steps, research training or weight
  restores during this repair. Temporary synthetic models keep the four-update cap.
- 63 core/config/runtime files remain byte-identical; CH3/CH4/CH5 full-tree hashes
  match the earlier task baseline (369/103/318 files). Preexisting user drafts
  and result assets remain excluded from staging.

Evidence: preparation.json (full input hashes), validation.json/synthetic.log,
clean_validation.json/clean_synthetic.log, integrity.json and frozen before/ files.
Linux should fetch the fix, generate a new validation directory and pilot root,
then explicitly run the existing pilot launcher. Keep the failed log and outputs.
