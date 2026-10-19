# 当前逐文件职责

下表只列当前可执行文件；历史来源/证据保留在provenance和verification的history内。

| 文件 | 职责 | 类/函数 |
|---|---|---|
| [chapter3_bser/candidate_generator.py](../chapter3_bser/candidate_generator.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `_local_peak_indices`, `_frontier_indices`, `_ordered_pool`, `_separated`, `generate_search_candidates`, `generate_standby_candidates`, `generate_candidates` |
| [chapter3_bser/config.py](../chapter3_bser/config.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `load_bser_phase1a1_config` |
| [chapter3_bser/controllers/path_tracker.py](../chapter3_bser/controllers/path_tracker.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `_TrackingState`, `PathTrackerSnapshot`, `PathTracker` |
| [chapter3_bser/controllers/state_provider.py](../chapter3_bser/controllers/state_provider.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `OnlinePlanningStateProvider` |
| [chapter3_bser/d2/allocator.py](../chapter3_bser/d2/allocator.py) | D2先验分配、导航、安全视图与审计记录 | `finite`, `EffectAllocator` |
| [chapter3_bser/d2/assembly.py](../chapter3_bser/d2/assembly.py) | D2先验分配、导航、安全视图与审计记录 | `prepare`, `build_controller`, `build_bridge`, `diagnostics` |
| [chapter3_bser/d2/contract.py](../chapter3_bser/d2/contract.py) | D2先验分配、导航、安全视图与审计记录 | `planner_protocol`, `enabled` |
| [chapter3_bser/d2/controller.py](../chapter3_bser/d2/controller.py) | D2先验分配、导航、安全视图与审计记录 | `D2SearchController` |
| [chapter3_bser/d2/logging.py](../chapter3_bser/d2/logging.py) | D2先验分配、导航、安全视图与审计记录 | `read_records`, `JsonlJournal` |
| [chapter3_bser/d2/options.py](../chapter3_bser/d2/options.py) | D2先验分配、导航、安全视图与审计记录 | `performance_options` |
| [chapter3_bser/d2/planning.py](../chapter3_bser/d2/planning.py) | D2先验分配、导航、安全视图与审计记录 | `locked`, `SafetyOccupancy`, `SafetyState`, `D2PlanningViews` |
| [chapter3_bser/d2/scored_controller.py](../chapter3_bser/d2/scored_controller.py) | D2先验分配、导航、安全视图与审计记录 | `EffectPolicy`, `ScoredWaypoints`, `EffectController` |
| [chapter3_bser/d2/solver.py](../chapter3_bser/d2/solver.py) | D2先验分配、导航、安全视图与审计记录 | `fixed_greedy`, `best_standby`, `solve` |
| [chapter3_bser/detection_model.py](../chapter3_bser/detection_model.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `_point_segment_distance`, `detection_probability` |
| [chapter3_bser/events/event_detector.py](../chapter3_bser/events/event_detector.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `EventDetector` |
| [chapter3_bser/events/event_types.py](../chapter3_bser/events/event_types.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `BSEREvent`, `EventDetection` |
| [chapter3_bser/events/trigger_policy.py](../chapter3_bser/events/trigger_policy.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `TriggerDecision` |
| [chapter3_bser/greedy_solver.py](../chapter3_bser/greedy_solver.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `solve_fixed_standby_greedy`, `solve_joint_greedy` |
| [chapter3_bser/hysteresis/cooldown.py](../chapter3_bser/hysteresis/cooldown.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `ReplanningCooldown` |
| [chapter3_bser/hysteresis/policy.py](../chapter3_bser/hysteresis/policy.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `ReplanningPolicy` |
| [chapter3_bser/integration/control_context.py](../chapter3_bser/integration/control_context.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `AgentAssignmentContextV1`, `ExecutorAssignmentContextV1`, `BSERControlContextV1` |
| [chapter3_bser/integration/guided_env.py](../chapter3_bser/integration/guided_env.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `GuidedEnv` |
| [chapter3_bser/integration/rmaddpg_bridge.py](../chapter3_bser/integration/rmaddpg_bridge.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `_vector3`, `_mission_phase`, `RMADDPGGuidanceBridge`, `get_tracking_targets` |
| [chapter3_bser/learning/base_runtime.py](../chapter3_bser/learning/base_runtime.py) | B2/B3模型、门控、上下文或Replay学习机制 | `seed_innovations`, `_make_base_env`, `_public_context`, `_observations`, `BaselineMissionRuntime` |
| [chapter3_bser/learning/boundary.py](../chapter3_bser/learning/boundary.py) | B2/B3模型、门控、上下文或Replay学习机制 | `boundary_features`, `BoundaryEncoder`, `BoundaryConditionedPolicy`, `BoundaryConditionedMADDPG`, `_contract_environment` |
| [chapter3_bser/learning/checkpoint.py](../chapter3_bser/learning/checkpoint.py) | B2/B3模型、门控、上下文或Replay学习机制 | `digest`, `write_json`, `validate_config`, `validated_output`, `fresh_source_identity`, `validate_source_identity`, `require_source_match`, `state_digest`, `_finite_state` |
| [chapter3_bser/learning/config.py](../chapter3_bser/learning/config.py) | B2/B3模型、门控、上下文或Replay学习机制 | `scenario_fingerprint`, `validate` |
| [chapter3_bser/learning/evaluate.py](../chapter3_bser/learning/evaluate.py) | B2/B3当前检查点评估、场景隔离与最终来源提交 | `preserved_rng`, `evaluate`, `main` |
| [chapter3_bser/learning/evaluation_records.py](../chapter3_bser/learning/evaluation_records.py) | 三方法统一指标与完成凭据/文件哈希 | `metric_row`, `finalize_evaluation`, `verified_rows` |
| [chapter3_bser/learning/execution_mask.py](../chapter3_bser/learning/execution_mask.py) | B2/B3模型、门控、上下文或Replay学习机制 | `ExecutionRepair` |
| [chapter3_bser/learning/gated_runtime.py](../chapter3_bser/learning/gated_runtime.py) | B2/B3模型、门控、上下文或Replay学习机制 | `execution_gates`, `RepairedMissionRuntime` |
| [chapter3_bser/learning/metric_ordering.py](../chapter3_bser/learning/metric_ordering.py) | B2/B3模型、门控、上下文或Replay学习机制 | `paired_ordering` |
| [chapter3_bser/learning/model.py](../chapter3_bser/learning/model.py) | B2/B3模型、门控、上下文或Replay学习机制 | `ExecutionRepair`, `RepairedMADDPG`, `RepairedBoundaryMADDPG`, `build_model` |
| [chapter3_bser/learning/provenance.py](../chapter3_bser/learning/provenance.py) | 精确来源、演进记录和本版本手动研究运行凭据 | `canonical_digest`, `source_files`, `verify_reviewed_changes`, `require_review`, `research_readiness`, `training_admission` |
| [chapter3_bser/learning/reference.py](../chapter3_bser/learning/reference.py) | D2零残差对照评估，无神经网络训练 | `ZeroResidualPolicy`, `evaluate_reference`, `main` |
| [chapter3_bser/learning/replay.py](../chapter3_bser/learning/replay.py) | B2/B3模型、门控、上下文或Replay学习机制 | `ExecutionReplay` |
| [chapter3_bser/learning/replay_base.py](../chapter3_bser/learning/replay_base.py) | B2/B3模型、门控、上下文或Replay学习机制 | `ExecutionSample`, `ExecutionReplay` |
| [chapter3_bser/learning/runtime.py](../chapter3_bser/learning/runtime.py) | B2/B3模型、门控、上下文或Replay学习机制 | `RepairedMissionRuntime` |
| [chapter3_bser/learning/scenarios.py](../chapter3_bser/learning/scenarios.py) | 生成场景并显式导出train/eval角色；物理任务保持不变 | `validate_manifest`, `export`, `main` |
| [chapter3_bser/learning/suite.py](../chapter3_bser/learning/suite.py) | 固定输入计划、独立seed任务、互斥锁和训练/评估调度 | `read`, `sha`, `prepare`, `load_plan`, `run_lock`, `training_receipt`, `run`, `summarize_suite`, `main` |
| [chapter3_bser/learning/summarize.py](../chapter3_bser/learning/summarize.py) | 已验证评估的逐seed统计、CSV/报告及配对排序 | `summarize`, `main` |
| [chapter3_bser/learning/train.py](../chapter3_bser/learning/train.py) | 母版当前Trainer、采样器、更新调度及检查点；新增独立来源/手动研究准入 | `Trainer`, `load_for_evaluation`, `main` |
| [chapter3_bser/objective.py](../chapter3_bser/objective.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `ObjectiveContext`, `ResponseDiagnostics`, `build_objective_context`, `cell_detection_probability`, `evaluate_objective`, `marginal_gain`, `expected_detection_probability`, `response_diagnostics` |
| [chapter3_bser/online/allocator.py](../chapter3_bser/online/allocator.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `_path`, `BSEROnlineAllocator` |
| [chapter3_bser/online/config.py](../chapter3_bser/online/config.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `execution_runtime_config`, `load_phase1b2_config`, `load_phase1b_config` |
| [chapter3_bser/online/controller.py](../chapter3_bser/online/controller.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `OnlineBSERController` |
| [chapter3_bser/online/execution_manager.py](../chapter3_bser/online/execution_manager.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `_path`, `ExecutionManager` |
| [chapter3_bser/online/mission_context.py](../chapter3_bser/online/mission_context.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `OnlineMissionContext` |
| [chapter3_bser/online/route_impact.py](../chapter3_bser/online/route_impact.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `RouteCheck`, `RouteImpactResult`, `RouteImpactEvaluator` |
| [chapter3_bser/online/state_cache.py](../chapter3_bser/online/state_cache.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `OnlineStateCache` |
| [chapter3_bser/online/types.py](../chapter3_bser/online/types.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `SearchAssignment`, `ExecutorAssignment`, `OnlineAllocation`, `CurrentAssignment`, `InitialBSERAllocation`, `BSERActionAssignment`, `OnlineStepDiagnostics` |
| [chapter3_bser/online/waypoint_manager.py](../chapter3_bser/online/waypoint_manager.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `WaypointManager` |
| [chapter3_bser/protocols/controller_factory.py](../chapter3_bser/protocols/controller_factory.py) | 当前复用的阶段/交接元数据，不是独立PRRAC网络 | `ExecutionVariant`, `PRRACRuntimeContract`, `build_prrac_online_controller` |
| [chapter3_bser/protocols/execution_diagnostics.py](../chapter3_bser/protocols/execution_diagnostics.py) | 当前复用的阶段/交接元数据，不是独立PRRAC网络 | `_runtime`, `_number`, `_integer`, `_vector`, `_mean`, `_sum_reward`, `_event_names`, `ExecutionEpisodeDiagnostics` |
| [chapter3_bser/protocols/stage_mapping.py](../chapter3_bser/protocols/stage_mapping.py) | 当前复用的阶段/交接元数据，不是独立PRRAC网络 | `PRRACStage`, `transition_phase_to_prrac_stage` |
| [chapter3_bser/protocols/team_environment.py](../chapter3_bser/protocols/team_environment.py) | 当前复用的阶段/交接元数据，不是独立PRRAC网络 | `PRRACTrainingEnv` |
| [chapter3_bser/protocols/transition_protocol.py](../chapter3_bser/protocols/transition_protocol.py) | 当前复用的阶段/交接元数据，不是独立PRRAC网络 | `PRRACTransitionMetadata` |
| [chapter3_bser/protocols/transition_schema.py](../chapter3_bser/protocols/transition_schema.py) | 当前复用的阶段/交接元数据，不是独立PRRAC网络 | `TransitionPhase`, `classify_transition_phase`, `Phase1CTransitionMetadata` |
| [chapter3_bser/rewards/reward_adapter.py](../chapter3_bser/rewards/reward_adapter.py) | 原奖励适配、团队均值与奖励日志 | `Phase1CExecutionRewardConfig`, `RewardAdjustmentResult`, `Phase1CExecutionRewardAdapter` |
| [chapter3_bser/rewards/team_objective.py](../chapter3_bser/rewards/team_objective.py) | 原奖励适配、团队均值与奖励日志 | `objective_identity`, `RewardAccounting` |
| [chapter3_bser/rewards/training_env.py](../chapter3_bser/rewards/training_env.py) | 原奖励适配、团队均值与奖励日志 | `Phase1CV2TrainingEnv` |
| [chapter3_bser/safety/allocators.py](../chapter3_bser/safety/allocators.py) | 公开地图路径安全、可达性与恢复策略 | `SearchPoolMixin`, `SafeJointAllocator` |
| [chapter3_bser/safety/failure_policy.py](../chapter3_bser/safety/failure_policy.py) | 公开地图路径安全、可达性与恢复策略 | `SearchFailurePolicy` |
| [chapter3_bser/safety/planning_refresh.py](../chapter3_bser/safety/planning_refresh.py) | 公开地图路径安全、可达性与恢复策略 | `start_status`, `topology_key`, `PlanningViews` |
| [chapter3_bser/safety/public_geometry.py](../chapter3_bser/safety/public_geometry.py) | 公开地图路径安全、可达性与恢复策略 | `_locked`, `_vector`, `PublicGeometry`, `_component_labels`, `_graph_hash`, `filter_planning_state` |
| [chapter3_bser/safety/runtime.py](../chapter3_bser/safety/runtime.py) | 公开地图路径安全、可达性与恢复策略 | `SearchController`, `SafetyMixin` |
| [chapter3_bser/safety/safe_guidance.py](../chapter3_bser/safety/safe_guidance.py) | 公开地图路径安全、可达性与恢复策略 | `_signature`, `SafetyBridge` |
| [chapter3_bser/types.py](../chapter3_bser/types.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `SearchCandidate`, `StandbyCandidate`, `CandidateGenerationResult`, `SolverResult` |
| [checks/test_learning.py](../checks/test_learning.py) | 当前实现的必要定向验证；不进入训练运行依赖 | `fresh`, `transition`, `sample`, `modified`, `snap`, `params`, `same`, `Constants`, `ActiveRLTests` |
| [checks/test_workflow.py](../checks/test_workflow.py) | 当前实现的必要定向验证；不进入训练运行依赖 | `scene`, `config`, `Workflow` |
| [cli.py](../cli.py) | 候选/目标、在线控制、状态、事件或任务衔接 | `main` |
| [core/algorithms/agents.py](../core/algorithms/agents.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_validate_adam_slot`, `validate_optimizer_state`, `validate_module_snapshot`, `validate_optimizer_snapshot`, `optimizer_transaction`, `DDPGAgent` |
| [core/algorithms/maddpg.py](../core/algorithms/maddpg.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `MADDPG` |
| [core/algorithms/misc.py](../core/algorithms/misc.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `soft_update_group`, `hard_update`, `onehot_from_logits`, `sample_gumbel`, `gumbel_softmax_sample`, `gumbel_softmax` |
| [core/algorithms/networks.py](../core/algorithms/networks.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `Identity`, `MLPNetwork` |
| [core/algorithms/noise.py](../core/algorithms/noise.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `OUNoise` |
| [core/communication/basic_communication.py](../core/communication/basic_communication.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_copy_target`, `FixedReliableHandoff` |
| [core/config/active_environment.py](../core/config/active_environment.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `build_ch3_config` |
| [core/config/ch3_constants.py](../core/config/ch3_constants.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | 包常量 |
| [core/env/mission_env.py](../core/env/mission_env.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `environment_kwargs_from_config`, `_array`, `_vectors`, `MissionCoreEnv` |
| [core/env/observation_contract.py](../core/env/observation_contract.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | 包常量 |
| [core/env/target_motion.py](../core/env/target_motion.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_nonnegative_int`, `_validate_finite_numbers`, `_vec3`, `_expanded_boxes`, `segment_aabb_first_hit`, `_boundary_first_hit`, `TargetState`, `advance_target_state`, `_plain_prediction_metadata`, `_prediction_context`, `_prediction_step`, `predict_target_state`, `simulate_target_trajectory`, `swept_relative_min_distance`, `solve_intercept_point` |
| [core/env/task_protocol.py](../core/env/task_protocol.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `protocol_identity`, `validate_task_config`, `require_protocol_output`, `closed_segment_aabb_first_hit`, `EpisodeOutcome`, `strict_terminal`, `episode_result` |
| [core/env/task_state.py](../core/env/task_state.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `MissionTaskState`, `AgentStateView`, `SearchExecutionState` |
| [core/env/uav_env.py](../core/env/uav_env.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_BaseUAVEnv`, `DummySpace`, `UAVEnv` |
| [core/mapping/map_module.py](../core/mapping/map_module.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `PheromoneWaypointPlanner`, `ProbabilisticTaskMapPlanner` |
| [core/mapping/path_planner.py](../core/mapping/path_planner.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `ObstacleAwareTaskMapPlanner`, `_online_cells`, `_online_segment_cells`, `_sampled_online_segment_free`, `_online_physical_time`, `OnlineUnknownMapTaskPlanner` |
| [core/mapping/planning_graph.py](../core/mapping/planning_graph.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_readonly`, `PlanningEdgeView`, `PlanningConnectorView`, `EndpointConnectorSet`, `PlanningGraphView`, `_flat`, `_diagnostic_snapshot`, `_topology`, `_component_labels_from_topology`, `_build_role_adjacency`, `_build_endpoint`, `_graph_hash`, `build_planning_graph` |
| [core/mapping/planning_state.py](../core/mapping/planning_state.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_readonly`, `GridGeometryView`, `TargetBeliefView`, `OccupancyBeliefView`, `PlanningAgentView`, `PlanningStateView`, `planning_state_sha256`, `_triple`, `extract_planning_state` |
| [core/mapping/travel_cost_service.py](../core/mapping/travel_cost_service.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_locked`, `PathQueryResult`, `SingleSourceTravelResult`, `TravelCostService` |
| [core/replay/ch3_buffer.py](../core/replay/ch3_buffer.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_resolve_device`, `CH3ReplayBuffer` |
| [core/scenarios/ch3_generator_impl.py](../core/scenarios/ch3_generator_impl.py) | 共享环境、动力学、目标/障碍、地图、MADDPG与Replay；本次未改动 | `_canonical_sha`, `_inside`, `_sample_point`, `_planner`, `_boxes_overlap`, `_sample_unknown_agents`, `_sample_unknown_obstacles`, `_unknown_trajectory_record`, `_validate_unknown_truth_reachability`, `_unknown_profile_obstacles`, `_build_unknown_profile_manifests`, `build_scenario_manifests` |
| [project_paths.py](../project_paths.py) | 候选/目标、在线控制、状态、事件或任务衔接 | 包常量 |
| [tools/run_checks.py](../tools/run_checks.py) | 独立来源校验或有界验证入口；不启动长训练 | `write`, `structural`, `regression`, `bounded_workflow`, `main` |
| [tools/verify_source.py](../tools/verify_source.py) | 独立来源校验或有界验证入口；不启动长训练 | `digest`, `verify`, `main` |

| [tools/validate_checkout.py](../tools/validate_checkout.py) | Git暂存字节、包完整性、提交/平台和有界工程验收；默认只打印计划 | `verify_index`, `verify_package`, `verify_clean_commit`, `run` |
| [checks/test_checkout.py](../checks/test_checkout.py) | 传输、缺失/篡改证据、误入模型及平台声明的7个负/正对照 | `Checkout` |
