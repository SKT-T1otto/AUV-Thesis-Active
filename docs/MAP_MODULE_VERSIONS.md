# 地图模块版本与对照基线

本表从 2026-10-20 开始记录 Active 的地图研发。所有实现、验证和未来实验都在本目录进行，不从旧仓库导入运行代码。版本记录不代表性能提升。

| 版本 | 内容与用途 | 当前状态 |
| --- | --- | --- |
| map_d2_repair_v2_20261020_v1 | 修改前 Active 的 D2 地图、候选、分配、安全、控制、配置与学习合同 | 已冻结。原来源标识 954f773a96b297ad8af3ec49eeabf969bf9b2326bdba67ea03c39d3044941f4f；可用于后续原地图对照 |
| map_public_interface_v1 | 默认关闭的公共扫描/步骤事务、去重和有效性验证、实际安装路线审计 | 第一批接口实现。开启时仍由原 D2 搜索分配；不是新地图算法 |
| coupled_search_core_v1 | 持续地图与目标位置/速度联合权重、首次发现签名、候选压缩/枚举及执行误差情景 | 核心第一版已实现，默认关闭；独立来源与机制证据，未做性能验收 |
| map_experiment_v1 | 旧/新地图 × ZERO/B2/B3 原生独立训练、评估、汇总与 Linux 启动 | 脚本已实现并经有限验证；Linux 实机与正式性能运行待用户明确执行 |
| 完整论文消融与跨地图诊断 | 简单修复组、完整地图消融、冻结策略跨地图、后续 RL 研究 | 尚未完成，不能与原生六组脚本及机制验证混同 |

当前核心说明见 [MAP_CORE_IMPLEMENTATION_20261020_v1](MAP_CORE_IMPLEMENTATION_20261020_v1.md)。下文“第一批”段落是冻结公共接口阶段的历史记录；当前核心记录追加于文末。

## 冻结内容

基线目录为 provenance/baselines/map_d2_repair_v2_20261020_v1/。

- source_and_evidence.zip：210 个文件的原字节快照，包含完整 139 项来源库存、原 active manifest、PACKAGE_SHA256、旧验证证据和文档。只作为冻结数据读取，不作为运行时 Python 依赖。
- baseline.json：每个归档成员哈希、原来源标识、Git HEAD 与冻结时状态、归档哈希。没有学习权重，D2 本来就没有学习权重。
- baseline_trace.json：改代码之前，以零残差策略采集的正常/严格碰撞两类短程轨迹。
- trace_receipt.json：以上轨迹的来源、哈希与预算；共 2 回合、5 个物理步，每回合最多 4 步，0 次优化器更新、0 次模型恢复。

原 provenance/active_manifest.json、provenance/rebuild_evolution.json、provenance/parent、provenance/history 与已有 verification 证据保持原字节。新接口用独立的 map_stage1_manifest.json、map_stage1_evolution.json 和 verification/map_stage1/ 记录，不把旧凭据改写为新版本结果。

后续需要运行冻结版本时，应在 Active/work 下从该归档建立明确的隔离实验副本，先核验成员哈希和原来源；不要把归档加到生产运行时的 sys.path，不要把新方法的结果标为原 D2，也不调用外部旧仓库。该步骤并未在本轮自动执行。

## 第一批实现的边界

原路径 core/mapping/map_module.py、path_planner.py、planning_state.py、planning_graph.py、travel_cost_service.py，以及 candidate_generator、objective、d2 求解器、所有 RL 网络/Replay/Trainer/奖励模块均保留。共享环境新增可选事件记录，原发现判据、动作与动力学、目标运动、建图融合、奖励和终止次序不变。

新接口按既有扫描复制数据，不再次调用真实射线查询。新步骤事件只含四体公开位置/速度、检查完成标记、发现状态和任务终止标签，没有隐藏目标位置/速度、真实障碍、流场相位或完整 scenario。观测适配器独立于环境对象，拒绝缺失时序、冲突重复、碰撞负证据和终止后的新事件。

已存在的 allocation 哈希没有完整覆盖路径，故另以完整实际安装路径、终点、角色、Hold 和可达性生成研究路线哈希。正常 tracker 进度变化不会被误记为路线替换。

原 28D/3D/124D、4 个智能体、gamma=.95、原奖励、网络、优化器与更新节奏保持。现有 B2/B3 方法身份不改。接口配置与基线学习配置分离，不向原训练配置中塞入新算法标签。

## 使用接口

configs/coupled_search_v1.json 默认 enabled=false。第一批只接受 mode=observe_only；请求尚未实现的耦合推断/新分配模式会明确拒绝。

默认只显示计划，不读场景、不建环境、不创建输出：

    python -m chapter3_bser.experiments.coupled_search_v1.reference --config configs/b2_mechanism.json --interface-config configs/coupled_search_v1.json --manifest <eval_manifest.json> --output <新的collision_terminal子目录>

如以后确需手动审计，先复制配置到新的文件并明确设 enabled=true，再加 --execute。该入口仅允许机制预算（最多 8 回合、每回合 32 步、总计 256 步），使用四体零残差，不加载模型。每次输出必须是新目录，包含 resolved config、合法公共事件、路线审计和完成记录。

它不是正式性能实验入口，输出明确标记 coupled_search_implemented=false、algorithm_changed=false、performance_passed=false。原 cli reference/evaluate/train 的入口和身份仍保留；本轮没有执行训练命令。

## 后续对照实验如何保持可比较

先完成实施说明的批次 2-5 及相应门槛，再锁定新算法版本和独立场景清单。地图对照建议命名 MAP_BASE、MAP_NO_REPEAT、MAP_SPATIAL、MAP_SIMPLE_COMBINED、MAP_FILTER、MAP_FULL，避免与现有 RL B2/B3 混淆。

第一轮地图贡献比较用同一零残差策略、相同物理任务/公开信息/安全与交接协议；记录算法种子、候选与查询预算、预测时域及 CPU 时间。压缩机制比较使用同一原始池和目标/执行样本；闭环实验按场景配对。执行者待命策略保持原机制，但其路径与扫描贡献仍须记录。

现阶段只建立可审计的基线和接口，没有验证新地图的 Found、发现时间、碰撞或超时改进。正式训练、旧权重兼容和性能结论仍需相应后续工作，不能由来源检查或短程等价检查替代。

## 本批验证记录

当前来源标识为 b31f45cb16d5ac3bd7586dcfe396b2f605758f32775f3f99fdb736cb5357d280，完整清单与精确逐文件演进分别位于 provenance/map_stage1_manifest.json 和 provenance/map_stage1_evolution.json。

- 60 项现有及新增检查全部通过，0 failure / error / skip；其中原有 38 项保留，新增 22 项覆盖事件与来源负对照。
- 独立文件验证器核验全部 160 项来源库存、基线归档和旧证据；剥离明确观测钩子后，两个环境文件与 base_runtime 的原有逻辑 AST 与冻结版本相同。
- 正常搜索与严格碰撞两场景，在关闭/开启采集的两种情况下，均与修改前轨迹逐步完全一致，核对内容包含观测、动作、物理状态、地图、奖励、终止、门控、引导和随机数状态。
- 另用两个回合验证首次发现、发现后的搜索者冻结、公开交接、时限与成功终止；发现后的步不重复发布发现检查，碰撞步负观测计数为 0。
- 本轮所有机制执行合计 8 回合、29 个物理步，每回合最多 8 步；没有训练任务、已有检查点恢复或提交/推送。既有学习机制回归仅使用临时新模型。

紧凑证据保存在 verification/map_stage1/。首次 standalone dry-run 测试的 Windows 临时目录清理失败及修正也单独保留。以上验证是本地、有限场景的接口证据，不是全状态空间证明、Linux 结果、历史 E0 或新地图性能结论。

## 核心第一版：coupled_search_core_v1

在改变算法前，另冻结 provenance/baselines/map_public_interface_20261020_v1/，235 个来源/证据文件，归档 SHA256 为 9b3a6e50c5ab22928a599421560408c75a2992e278e419856e1d4b9366b4f68d。原 D2、Stage-1 manifest 和 verification/map_stage1 的字节均未修改。

当前独立来源为 provenance/map_core_manifest.json 与 map_core_evolution.json；人工审阅来源标识 8e8493417642d9c99b55c8ab1b6ecb6e48d56c4acec77dee6911c9869751d15d，共 179 项库存。只改授权的地图研究分支、组合入口、公开常量读取和验证工具，原网络、奖励、训练和控制物理合同保持。

85 项检查全部通过，无跳过。核心闭环覆盖正常搜索/重规划、近目标发现与公开交接、严格碰撞回退；默认关闭分支与冻结 D2 轨迹逐步相同。当前轮合计 8 回合、107 个物理步，每回合最多 32 步；未启动训练，未恢复已有模型。核心闭环记录来自最终审计字段完善前的候选版本，保留原来源身份；最终版本另验证终点审计与默认关闭等价性，不重标旧结果。

完整证据与范围限制见 verification/map_core_v1/。连续地图几何先验是扫描条件下的近似补全，默认数值未调优；碰撞夹具的假设生成耗尽预算时确实回退。此版本不是完整 PDF 的消融/实验交付，也不是新地图性能提升证明。

## 原生地图完整系统实验脚本：map_experiment_v1

地图核心改变前的来源 8e849341... 已另冻结在 provenance/baselines/map_core_20261020_v1/，300 个文件，归档 SHA256 为 752ef1cd107d4d2a8f5d948660ca157b325b873af805fc130dbc5cfff86e7209。所有原 manifest 和 verification 证据保持原字节。

本轮只扩展训练/评估装配、地图与检查点身份、冻结任务顺序、六组调度、统计和 Linux 验收；原地图核心、环境与 B2/B3 学习更新不改。新独立来源为 provenance/map_experiment_manifest.json 与 map_experiment_evolution.json，来源标识 efb7805f7e166137a16cd20f869cd602fd294dea715fb197d528df70499b294c，193 项库存。

112 项检查通过，0 failure/error/skip。八个实际机制回合使用 OLD/NEW 的 B2/B3 四个临时新模型：各训练 8 步、评估严格碰撞 1 步，共 36 步；每模型 4 次完整多智能体更新。配对初始权重一致，NEW 在正常训练夹具实际参与规划，评估策略状态保持不变。评估复用当轮内存模型，没有恢复检查点文件。六组完整汇总另有纯数据合同检查，ZERO 原逻辑与此前证据保留，不把四组机制结果重标为正式六组性能数据。

脚本包括 20 任务的正式配置和开发配置；所有执行仍需 --execute，正式六组运行还需当前固定提交的 Linux 验收。Bash 语法检查已通过；Windows 沙箱内的 Git Bash 运行初始化失败，随后只读沙箱外语法核验成功。当前未进行 Linux 实机测试、正式训练、提交或推送。完整用户流程见 MAP_EXPERIMENT_LINUX_20261020_v1.md，证据见 verification/map_experiment_v1/。
