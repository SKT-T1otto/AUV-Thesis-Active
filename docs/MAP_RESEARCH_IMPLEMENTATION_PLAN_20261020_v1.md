# 地图研究方案在 Active 中的具体实施建议

日期：2026-10-20。适用目录：E:/gym/code/WORKSPACE/AUV-Thesis-Active。

本文是对现有代码的只读审计与拟议改动清单，不是算法实现记录。本轮只新增本说明，不修改运行代码、配置、历史证据，不创建环境回合、不训练、不恢复 checkpoint、不提交或推送。下文文件和接口标为“新增/修改”的，均是后续建议。

依据是地图设计 PDF（2026-10-17，16 页），并查看同目录 RL 方案的可编辑参考（2026-10-18，方案 1/2/4）。地图方案第 4-11 页的关键公式另经页面渲染核对。本文不重新核验文献新颖性，也不把 PDF 内历史实验数字当作 Active 当前实验结果。PDF 内旧仓库路径、旧行号与执行安排仅作资料；本次用户请求和 Active 的 AGENTS.md 决定工作范围。

**建议：保留原 D2 repair-v2，新增默认关闭的 coupled_search_v1 研究分支。先完成合法观测事务和耦合信念，再实现首次发现签名、候选压缩与团队枚举，最后增加执行误差情景。先以四体零 RL 残差隔离地图贡献，后续单独推进 RL。**

## 1. 当前代码能复用什么，具体缺什么

| 现有位置与函数 | 当前实际行为 | 对方案的影响 |
| --- | --- | --- |
| core/env/uav_env.py::_sense_and_update_shared_map，1476 行 | 四体当前位置、26 个射线方向、距离和 hit_mask 已在原建图前形成；1489 行调用一次 integrate_obstacle_scan | 原始合法扫描载荷已经存在，应复制这次载荷；不需要再次 raycast，也不需要修改融合算法 |
| core/env/uav_env.py::reset，1155 行 | 显式 reset 时先做第 0 步共享扫描 | 采集接口必须在 reset 前启用，否则会丢掉第一批地图证据；第 0 步扫描不自动等于完成了发现检查 |
| core/env/uav_env.py::step，1549-1571 行 | 智能体运动、严格碰撞提前返回、共享扫描、目标运动、原信念更新、swept 发现检查依次执行 | 事件必须描述实际执行顺序；“found=False”不能单独作为本步负观测 |
| core/mapping/planning_state.py::PlanningStateView，60 行 | 有目标位置边际、占据边际、公共位置速度与规划图 | 缺原始扫描事务、位置-速度联合假设、发现检查有效性；不能从两个边际还原精确联合后验 |
| chapter3_bser/controllers/state_provider.py::snapshot，64 行 | 非完整刷新步仅更新时间、任务标记和智能体字段，地图/目标边际继续使用缓存；默认完整刷新间隔为 20 步 | 私有信念应逐物理步消费事件，不能只在 provider 刷新或规划触发时更新 |
| core/mapping/path_planner.py::predict_belief_motion，1098 行 | 位置栅格扩散；occupancy_constrained 模式每次乘一次当前自由概率，再归一化 | 这是重复占据加权的明确代码位置；应在独立简单修复对照中去掉重复加权，不能直接改掉原基线 |
| chapter3_bser/candidate_generator.py::_ordered_pool，48 行；generate_search_candidates，66 行 | 同概率按索引排序；每体遇到 K 个可行点就停止，当前配置 K=4 | 近均匀信念存在索引前缀偏置的机制基础；当前源码审计并未重新证明 PDF 中的历史场景统计 |
| chapter3_bser/d2/solver.py::solve，30 行 | D2 先按静态搜索效用做贪心分配，再选 executor standby | 与“3 个搜索者各 4 条代表路线，枚举最多 64 组”不同，需新增求解器 |
| chapter3_bser/objective.py::cell_detection_probability，57 行 | 静态栅格探测模型，组合 1-product(1-p) | 不是共享目标轨迹上的团队首次发现；新分支需单独的签名与效用实现 |
| chapter3_bser/safety/safe_guidance.py::SafetyBridge.compile_guidance，52 行 | 可以换掉规划路线，或进入 Hold；实际 guidance 可能不同于上层 allocation | 必须在安全编译和安装后记录实际路线，原预测随替换失效 |
| chapter3_bser/online/types.py::OnlineAllocation.allocation_sha256，48 行 | 哈希覆盖候选标识、终点、分数等，未完整覆盖路径点 | 不能只用 allocation_sha256 判定路线未改变；需在研究审计中另建实际路线哈希，不修改原哈希含义 |
| core/env/uav_env.py::_compute_waypoint_prior_acc，662 行；_apply_agent_dynamics，713 行 | 先验、残差、加速度限幅、阻尼、速度限幅、位置积分等都在环境内 | Active 没有 PDF 所提的独立 Vehicle 类；执行预测以这些函数为实际语义依据 |

当前关键调用链为：

    learning/base_runtime.py
      -> d2/assembly.py
      -> EffectController + EffectAllocator + D2SearchController
      -> SafetyBridge -> PathTracker -> GuidedEnv.install_guidance
      -> UAVEnv.step

新研究分支接入这一链路的上层搜索分配和旁路观测接口。公开建图、安全检查、物理控制、可靠交接、发现后冻结与终止仍沿用当前实现。

## 2. 拟新增目录与代码职责

研究主体全部放在 chapter3_bser/experiments/coupled_search_v1/。表中方法名为拟议接口，不是已存在 API。

| 新增文件 | 主要类型/函数 | 具体增加的代码 |
| --- | --- | --- |
| __init__.py、config.py | CoupledSearchConfig / validate_research_config | 默认关闭、只支持单目标与三个搜索者；校验先验、时域、样本数、查询次数与超时预算；独立 RNG，不消耗主环境/策略随机流 |
| types.py | PublicStepEvidence、JointBeliefSnapshot、RouteProposal、ForecastBundle、SignatureBundle | 不可变数据结构；携带物理步、数据版本、来源哈希、有效性和失败原因；RouteProposal 包含完整路径与唯一 route_id |
| public_observations.py | PublicObservationAdapter.ingest / consume_step | 从公共事件接口取数、按事件 ID 去重、检查步序和完整性；输出地图证据与有效负观测；禁止接受完整 scenario 字典 |
| geometry_hypotheses.py | GeometryHypothesisBank.initialize / condition / replenish | 生成和维护持续几何假设；公开扫描相容性检查；有界补提议；保留几何身份；未知空间先验必须预先声明 |
| joint_belief.py | CoupledBelief.initialize / update / forecast | 持续地图内的位置-速度假设；地图证据与未发现证据分开处理；两层权重、ESS、模式数、失效与必要历史重放 |
| region_proposals.py | propose_regions | 根据未来信念的空间和时间模式生成区域；加入少量原前沿和空间分层提议；消除同概率索引前缀偏置 |
| route_pool.py | build_route_pool | 每体有界原始路线池；复用 TravelCostService、公共安全几何；保留可行原参照；少量中间点绕行；查询总数也有上限 |
| execution_forecast.py | NominalExecutionForecaster / ExecutionScenarioBank | 先实现零残差名义跟踪，再加入公共执行误差形成的少量连续联合情景；每条路线共用情景 |
| discovery_signature.py | first_hit_times / build_signatures | 对共享目标轨迹和执行情景逐步用原 swept helper 检查首次命中，输出 f[agent,route,target_sample,scenario] |
| candidate_compression.py | select_representatives | 保留可行参照路线；按 PDF 式(9)的任务距离选代表，记录实际覆盖半径、是否达标及预算停止原因 |
| allocator.py | CoupledSearchAllocator.allocate / allocate_partial / rescore | 压缩后团队枚举；保留未受影响搜索者；统一快照与样本重评分；沿用 executor 待命及发现后分支 |
| controller.py | CoupledSearchController | 复用现有事件、冷却、稳定化与恢复链；对新分支实现同快照重评分、平分时保留原分配；不逐物理步强制重规划 |
| lifecycle.py、audit.py | PlanLifecycle / ResearchAudit | 跟踪预测、安装、实际执行、替换/结束；记录实际路径哈希、Hold、误差、耗时、查询与回退 |
| assembly.py | prepare / build_controller / build_bridge / diagnostics | 组装新研究对象；复用 D2PlanningViews、D2SearchController 与 SafetyBridge；研究依赖不进入 core |
| reference.py | research reference runner | 独立地图研究入口，只用 ZeroResidualPolicy；默认打印计划，显式 --execute 才运行；输出新的方法身份和目录 |

另外新增 chapter3_bser/planner_factory.py 作为小型装配分派器：关闭时委托现有 d2.assembly；开启时延迟导入 coupled_search_v1.assembly。不在 core 反向导入 Chapter 3。避免复制整份基线 runtime 或在原文件里堆满分支。

## 3. 现有文件具体怎么改

### 3.1 最小共享接口：只输出已发生的公共事件

**新增 core/env/public_events.py；修改 core/env/uav_env.py 与 core/env/mission_env.py。**

core 只定义与研究算法无关的不可变事件结构和可选事件缓冲，不放粒子、候选或研究评分。建议接口为 enable_public_event_capture(episode_id)、read_public_events(after_sequence)，名称最终以实现审查为准。关闭时不采集、不复制扫描数组、不改变随机流；开启后也不调用研究回调驱动环境。

uav_env.py 的具体落点：

1. reset 清理当前回合事件序列；采集开关由 MissionCoreEnv 在显式 reset 前打开，捕获 1155 行产生的第 0 步扫描。
2. _sense_and_update_shared_map 在现有 integrate_obstacle_scan 成功后，复制已经使用过的 origins、directions、distances、hit_mask，并记录所属物理步、传感来源及顺序号。原 integrate 调用仍只有一次，四体扫描仍按现有批次融合。
3. step 在动作执行前后保存允许公开的智能体位置/速度；在目标运动和原 _maybe_detect_swept 实际完成后，发布完成标记和 found 状态迁移，不发布隐藏目标坐标或速度。
4. _collision_terminal_transition 路径发布“本步发现检查未完成”的结束标记。不能因为 step_count 已增加且 found=False，就构造普通未发现事件。
5. 成功或时限终止步若确实完成了扫描/发现检查，仍保留这些已发生的事件；之后不再规划。

建议最少字段：episode_id、physical_step、sequence、source_id、kind、payload_hash、scan_payload、agent_start/end、velocity_start/end、target_transition_completed、discovery_check_completed、found_before/after、episode_terminated、terminal_reason。终止原因只给任务协议允许的标签，不泄漏真实障碍几何或内部碰撞对象信息。

唯一事务 ID 使用“回合 + 物理步 + 来源 + 类型 + 批内序号”。同一 ID 重读不得再次更新；同一 ID 载荷不一致直接报错。不同物理步的真实新扫描不能仅因载荷相同就被误删。快照重读则不是新扫描。

mission_env.py 只转发公共接口与公开车辆常数。PlanningStateView 的现有结构及 planning_state_sha256 不必为此改动，28D 观测也不增加地图、粒子、事件或时间维度。

如果原始扫描接口尚未完成，可先做占据快照 logit 差的明确近似验证；必须用内容差，不只看 map_revision。不能把该近似当作正式射线似然，也不能同时把原扫描与其融合后快照重复计为两份证据。既然当前原始载荷位置清楚，推荐直接完成上述最小接口。

### 3.2 Runtime 接入：按物理步更新，不遗漏终止步

**修改 chapter3_bser/learning/base_runtime.py；新增 planner_factory.py。**

| 当前落点 | 建议改动 |
| --- | --- |
| __init__ 内 assembly.prepare、build_controller、build_bridge，约 60/80/82 行；summary 的 diagnostics | 通过新 planner_factory 分派。无研究配置时调用同一原 D2 实现，不改原 d2.assembly 内的算法 |
| 构建 base 后、self.env.reset 前，约 62-70 行 | 仅开启研究分支时启用公共事件采集，绑定 episode_id，建立独立研究随机流 |
| provider.initialize 后、controller.initialize 前，约 76-81 行 | 消费第 0 步地图证据并初始化私有耦合信念；不得把 reset 后 found=False 当成负观测；避免把已融合这些扫描的旧边际再当独立证据融合一次 |
| self.env.step 后、if self.terminal 前，约 119-120 行 | 消费本次已发生事件，更新耦合信念与公共执行误差。终止步也进入审计；只有非终止分支才安装新规划 |
| compile_guidance / install_guidance / refresh_observation_after_guidance 后，约 83-85、127-129 行 | 记录最终安装上下文、路径及 tracker 状态，结束被替换的旧预测；不能只记录 allocator 提案 |
| close 与 summary | 关闭研究状态、输出诊断身份；默认关闭时不产生额外研究输出 |

这些是可选组合接口，主体物理步、动作门控、奖励包装和更新调度保持现有次序。未来改源码后需要新的来源演进和等价证据，不能继续声称旧源码字节或 AST 未变。

不把逐步信念更新塞进 OnlinePlanningStateProvider.snapshot，也不为“更新信念”每步重建完整 planning_graph。规划触发时使用现有 provider/安全刷新机制获得可用公共图；私有证据时间和规划快照时间分别记录。若需要额外图刷新，作为显式研究计算开销记录，并给对照相同刷新预算。

### 3.3 分配器：覆盖搜索部分，复用待命与交接

**主要新增 allocator.py、controller.py；原 d2/allocator.py、d2/solver.py、objective.py 保留。**

CoupledSearchAllocator 可继承 EffectAllocator 复用既有 executor 处理和分配数据结构，但必须覆盖搜索池生成、_proposal、allocate_partial、rescore 的研究实现；不能只覆盖 _proposal，随后又被原 rescore 改回静态分数。

局部重规划时，未受影响搜索者仅有“当前剩余路线”一个固定选项；对受影响搜索者生成新候选。所有旧路线先按当前位置裁切并检查当前公共可行性，不能直接拿包含过去起点的完整路径与新路线比较。

Executor 第一版继续使用原 standby_pool、执行分配、公开交接、拦截与安全处理机制。为减少归因混淆，standby 的候选与评分读取原公开地图/原信念，不把私有后验灌进 executor。搜索路线变化仍可能经原 standby 评分间接改变 executor 行为，必须记录待命目标、行程及其扫描贡献；这不等于 executor 轨迹保证相同。

target_found 后立即停止新搜索决策，调用原父分支，保持搜索者冻结与执行者接管。新地图方法命名为独立搜索规划身份；它使用零残差时也不能冒充原始 D2 数据。

控制器保留原事件检测与冷却；新旧分配只在同一私有信念、同一执行样本、同一规划快照下比较。D2 当前 EffectPolicy 会接受冷却后可用提案；新研究分支如果加入“相近收益保留原路线”的规则，应在自己的 policy 内显式实现并记录，不能宣称继承 D2 就自动具备这条规则。

WaypointManager 可能稳定化并保留部分旧路线。其结果必须用新效用重新评分，沿用现有 ScoredWaypoints 的时机，调用研究 rescore；安全层之后若又换路，另行使该预测失效或重算。

### 3.4 入口、配置与身份

新增 configs/coupled_search_v1.json，作为与原 B2/B3 配置分离的研究扩展配置。至少声明 enabled=false、运行模式、单目标、几何/速度先验来源、R/N/Q/H、每体原始/代表路线预算、路径查询上限、计算超时、重放历史上限、beta/rho/lambda_cost、成本归一化尺度、独立随机种子、回退规则及审计目录。

12 条原始路线压缩到 4 条是 PDF 给出的预算起点；4^3=64 是三个搜索者全体参与时的最大组合数，不是性能最优参数。R/N/H/Q 尚无项目验证，先从小型机制配置递增；名义版本 Q=1，beta=0、rho=0 可作为易解释对照。成本系数与归一化尺度应单独冻结，不能每个候选池各自 min-max 后比较新旧方案。

研究入口建议先使用完整模块路径：chapter3_bser.experiments.coupled_search_v1.reference；暂不需要修改 cli.py。它读取现有修复配置作为基础控制合同，再单独验证研究配置。原 learning/config.py 对 planner_protocol=d2_v1 等约束保留，不直接把新规划器伪装成该协议放行。

最终输出分别记录 base_control_protocol、search_planner_id、map_variant、policy_identity、源身份、resolved config、先验身份和种子。地图 PDF 的 B0/B1/B2/B3 是地图对照编号，与现有 RL B2/B3 无关；实现中建议使用 MAP_BASE、MAP_NO_REPEAT、MAP_SPATIAL、MAP_SIMPLE_COMBINED、MAP_FILTER、MAP_FULL 等不冲突名称。

## 4. 三部分算法具体怎么写

### 4.1 持续地图-目标联合假设

私有状态至少保存：R 个持续几何 M_r、地图权重 alpha_r、每图 N 个 (position,velocity) 假设、图内权重 omega_rn、上一物理步、已消费事务序列、必要公共历史与诊断。总轨迹权重为 w_rn=alpha_r*omega_rn。

初始化优先从显式冻结的公共位置/速度先验和第 0 步扫描建立状态。如果选择旧位置边际作初始化近似，必须声明它已经包含哪些证据，不能再重复同一批扫描；后续禁止每步融合旧 target_belief。

传播直接调用 core/env/target_motion.py::advance_target_state，传入自己构造的 TargetState 和 M_r 的假设障碍。bounds、dt、clearance、反射上限取公开合同。不能传 env.target_state、target_initial_velocity、ground_truth_obstacles，也不能把 scenario_id、场景种子编码成隐含真值来源。目标不额外增加流场或随机转向。

每步只对实际新扫描计算地图相容性 L_M,r。确定性扫描的兼容判断必须声明几何离散化容差；现有粗体素 known_free 与原始命中可能并不一致，不能同时把两者当成完全精确、互不相关的硬约束。正式模型优先保留原始射线约束，快照只作已有建图的近似摘要；任何冲突、近似和拒绝原因都记录。

对实际完成的未发现事件，用各搜索者公开步段和每个目标假设的步首/末位置计算 D_i；L_N=product_i(1-D_i)。这里是同一个目标假设上的确定性团队事件，不用再引入搜索者条件独立假设，也不沿用旧高斯负观测的漏检 floor 掩盖全灭。

实现时最容易遗漏的是图内负证据的归一化常数也影响地图权重：

    omega_raw[r,n] = omega_prior[r,n] * L_N[r,n]
    Z_N[r] = sum_n omega_raw[r,n]
    alpha_new[r] proportional to alpha_old[r] * L_M[r] * Z_N[r]
    omega_new[r,n] = omega_raw[r,n] / Z_N[r]  (Z_N[r] > 0)

只归一化 omega 而遗漏 Z_N 对 alpha 的作用，会丢掉 PDF 所要求的“未发现反过来改变地图假设权重”。数值稳定可用 log 权重，但零质量仍必须被识别为零。

退化时先检查时序、去重与探测语义，再在冻结合法先验内有界补提议，并重放必要公共历史。补新图后直接沿用旧目标状态是不一致的。历史不足、预算耗尽或全部假设失效时，输出 estimator_valid=false，回退既有 D2 搜索与安全链；不伪造有效后验，不回写原占据图。

### 4.2 区域、路线、签名压缩与枚举

未来轨迹每个规划时点只传播一次，各路线共享。区域提议可用未来位置的时间加权边际，但只用于找区域；同一个目标出现在多个未来时刻不能算成多个独立目标。

原始池按“未来主要模式 + 少量前沿 + 当前可行参照 + 少量空间分层/不同通道”构造。多段路线经由中间点查询拼接，去掉重复连接点，核对起终点和连续性，再经当前 PublicGeometry 检查。私有采样地图用于目标传播，不得作为已观测安全几何让未知路线被宣称安全。

同时约束已接收路线数与总查询数，失败查询也计预算。TravelCostService 默认返回一条优选路径，不等于已经提供 K 条替代路径；中间点与分层提议由新 route_pool 明确实现。保留参照仅限其当前仍可行，不能为凑齐 K 伪造路线。

用原 swept_relative_min_distance、原传感半径和 detect_eps_bias，从未来第 1 步寻找首次命中 T；无命中为 0，有命中 f=exp(-beta*dt*T)。离散步号与物理秒必须分开。环境当前发现检查使用每个物理步的目标首末位置段；即使反射内部有折线路径，也应先匹配原 helper 的现行语义，不能悄悄改成另一种探测规则。

压缩距离按 PDF 式(9)：各执行情景中目标加权签名绝对差的最大值，加一个现有规划成本差。保留可行参照，再用有界最远代表选择；停止后重新计算每体实际 epsilon_i。达到 K 不代表达到误差目标。候选仍须在筛选样本之外的独立复核样本上检查。

团队枚举对每个共享目标与执行情景取三条路线签名的 max，计算：

    F_e(S) = sum_l w_l * max_(i,a in S) f[i,a,l,e]
    J(S) = (1-rho)*sum_e nu_e*F_e(S) + rho*min_e F_e(S)
           - lambda_cost*sum_(i,a in S) C_i(a)

beta=0 时另报期限内发现概率；beta>0 的 J 不能直接写进名为 detection_probability 的字段。新分支为 OnlineAllocation.detection_probability 计算未折扣的有效概率，objective_value 保存 J；原静态指标、执行响应诊断与研究预测指标分名保存。

PDF 式(10)目前是待证明的有限池命题。实现检查可枚举原池与压缩池，报告真实模型内损失和 epsilon_i；它不能替代正式证明，也不能推出真实环境性能界。联合可行性约束若加入，须重新检查代表替换条件。

### 4.3 名义跟踪与执行误差情景

先在 execution_forecast.py 中建立独立局部预测状态与 PathTracker 实例，不推进 live bridge 的 tracker。继续当前路线时复制必要的进度；新路线按 PathTracker 的原始规则初始化。当前 tracker 每次调用至多推进一个路径点，不能改成 while 跳过所有近点。

按 _compute_waypoint_prior_acc、_actions_to_residual_acc、_apply_agent_dynamics 的顺序实现名义预测：导航先验、轴向尺度、先验/残差比例、两级限幅、阻尼、速度限制、位置积分和边界。参数取实际 resolved config 和公开规格，不能误用构造器默认残差比例；Active 配置为搜索 0.20、执行 0.15。

不调用 live env.step 预测候选，也不读取真实流场相位或未来残差。第一阶段使用四体零残差的名义模型；对未知扰动形成的偏差作为预测误差报告。之后用已经执行的公共速度/位置与预测差构造少量时间连续联合情景，区分共同分量与个体分量。它是合成执行误差，不冒称真实洋流。

未来所有候选共用执行情景与权重。目标几何假设和执行情景按 w_l*nu_e 组合是首版建模近似，若误差与区域相关需条件建模；它不是任意真实扰动下的鲁棒保证。

生命周期记录 proposed_route_hash、installed_route_hash、起始物理步、预测时域、有效区间、替换原因。哈希覆盖完整路径、终点、Hold/reachable 状态与路线版本，不只用原 allocation 哈希；正常沿同一路线推进 tracker 进度不自动视为换路。高层换目标、安全层替换、Hold、发现、终止时结束旧预测。无法描述的未来恢复不虚构收益。

## 5. 哪里要替换或减少，哪里保留

| 逻辑 | 新研究分支如何处理 | 原基线如何处理 |
| --- | --- | --- |
| 每步对同一占据边际重新乘权 | 新私有信念只消费新事件；简单修复对照保留扩散但去掉重复占据乘权 | 原 path_planner.py 保留；原建图与旧信念继续运行，供基线/原 standby/回退使用 |
| 只维持目标位置边际 | 增加持续地图内的位置-速度假设与轨迹 | 不删除原 belief_map |
| 同分按低索引取前 K 点 | 新 region_proposals 做空间分层、模式和时间多样性；独立简单修复对照仅改平分/截断规则 | 原 candidate_generator.py 不直接改 |
| 仅按端点/几何距离去重 | 限制原始池后，用首次发现效用签名压缩 | 原候选与几何方法保留为对照 |
| 静态概率与贪心求解 | 新分支不调用原静态效用作为搜索目标，改为轨迹签名与小规模枚举 | 原 objective.py、d2/solver.py 不删不改 |
| 每组合重新传播目标或预测路线 | 传播与逐路线签名先共享缓存，枚举只组合已算签名 | 无需影响原实现 |
| 每步重新规划、无界失败查询 | 不增加；沿用触发与冷却，有界查询/补提议/历史重放 | 原运行调度保留 |
| 预测时重复调用真值射线/复制 live 环境跑候选 | 不增加；仅用公共事件与局部预测模型 | 原真实扫描仍是唯一来源 |
| 运动、发现、碰撞、交接、终止、奖励、28D/3D/124D、网络与优化器 | 全部复用原行为 | 全部保留 |

本阶段没有需要从 Active 物理删除的旧运行模块。减少的是新分支里的重复证据、冗余候选与重复计算，不是删除对照、历史证据或保护性检查。

## 6. 建议拆成五批实施，每批单独可核验

| 批次 | 本批交付 | 验证与继续条件 |
| --- | --- | --- |
| 1 公共事件与装配接口 | public_events、环境事件出口、factory、默认关闭分支、read-only adapter | 第 0 步扫描、普通步、严格碰撞、时限步、去重、无真值泄漏；关闭前后相同动作下观测/状态/奖励/done/RNG 一致 |
| 2 耦合信念及简单修复对照 | 几何/速度先验、joint_belief、共享轨迹、MAP_NO_REPEAT / MAP_SPATIAL / 组合对照 | 无新地图证据不重复改变地图权重；双反射模式、负观测、两层质量、全灭与有界回退；未通过不接管路线 |
| 3 首次发现签名与名义执行 | Q=1 的 execution_forecast、签名、固定终点影子评分 | 原物理/探测语义匹配、目标样本共享、首次质量+存活质量=1、重复发现不重复计数；先观察模型误差 |
| 4 候选压缩与枚举接入 | route_pool、compression、allocator/controller、安装生命周期 | 同池枚举核对、epsilon 与模型损失、局部重规划冻结、稳定化后重评、安全替换/Hold 失效、预算回退 |
| 5 执行情景与有界闭环 | 连续联合情景、独立复核、日志汇总 | 公共误差来源、样本一致、短期 ETA/横向误差可解释；再评价地图机制与简单修复的差别 |

建议检查文件为 checks/test_public_events.py、test_coupled_belief.py、test_discovery_signatures.py、test_candidate_compression.py、test_coupled_search_integration.py；纯数据测试不建环境，必要机制闭环总量控制在 AGENTS.md 的 8 回合 x 32 步以内，不训练模型。若后续 RL 验证需要更新，每个临时模型最多 4 次完整多智能体更新。不要把全套机制上限按测试函数重复领取。

每次验证新建 work/map_research/<unique_run_id>/，不覆盖既有目录。正式研究训练仍需当时有效的来源与验证凭据，以及用户显式 --execute 安排；本实施建议不授权启动它。

checks/test_learning.py、test_workflow.py、test_checkout.py 和 tools 内现有相关检查继续保留。新增检查必须显式加入检查清单；不得降低“无 skip、库存一致、来源匹配”等原有要求。

## 7. 来源演进是实际接入的一部分

当前 learning/provenance.py::source_files 会纳入 core、chapter3_bser、configs、checks、tools 中新增的 py/json。即使开关关闭，仅新增研究源码也会改变来源库存，旧 active_manifest 与旧 readiness 不能直接覆盖新版本。

后续每批先保存并引用不可变父清单，再建立新的版本化演进记录与独立验证结果。记录逐文件 before/after SHA256、改动原因、默认关闭等价范围、开启后的算法身份。必要时扩展现有验证器，使其验证新的父子链和版本化 receipt，保持旧证据可追溯；不得通过忽略新文件、改写 parent/history、绕过 training_admission 或把旧测试结果绑定新哈希放行。

PACKAGE_SHA256.json、当前 active manifest 与 readiness 的新版本只应在真实检查完成后生成；旧版本先留存原字节。本轮没有改动它们。原 D2/B2/B3 checkpoint 不重命名为新地图方法；来源改变后的旧权重使用需要独立、可审查的兼容方案，不自动加载。

PDF 所列旧 bottleneck/shadow/safe_search_v2/motion.py、旧 core/runtime/engine.py、旧 baselines/common/runtime.py 与 HGR 入口不在本轮可复用清单。不从旧仓库补拷贝，也不通过 sys.path 注入。文中历史 20 场景、420 条记录、E0 与创新结论均不在这里重建或转移。

## 8. 与后续 RL 方案的衔接

地图与 RL 分开推进。地图分支输出仍是原 OnlineAllocation / BSERControlContextV1，Actor 仍输出每体 3D 归一化残差。私有联合后验和完整假设地图不进入 Actor，也不为它们扩充 28D 或 124D。B3 现有公共上下文编码另按其原边界接口保留。

后续 RL 方案建议放在另一个独立目录 chapter3_bser/experiments/event_value_calibration/：snapshot_adapter、event_observer、intervention_sampler、branch_runner/labels、branch_cache/losses、trainer/audit。方案 1 先用真实成对单步干预校准原 Q 的动作价值差；方案 2 的四支交互及方案 4 的事件时域后置。

Active 对应关系：原 Actor/Critic 在 core/algorithms/maddpg.py 与 chapter3_bser/learning/model.py，B3 在 learning/boundary.py，Replay 在 learning/replay.py，真实决策闭环在 learning/base_runtime.py，更新调度在 learning/train.py。未来先用独立子类/损失扩展复用这些机制，不把旧文档提到的历史目录重新导入。

未来闭环快照必须在 guidance 安装并刷新观测后、下一次 Actor 决策前建立；包含环境、provider 缓存、controller、waypoint manager、SafetyBridge overrides、PathTracker、交接状态、动作门控、奖励包装、研究信念、事件读取游标、独立 RNG 和预测生命周期。只恢复位置速度或只 deepcopy 环境都不够。确定性前缀回放可以作为先期方案，但其全部成本要计入。

公共扫描事件只保存合法信息；RL 模拟器恢复所需的完整隐藏状态属于独立的训练仿真设施，不能混入规划事件或部署输入。

RL 初版保留 gamma=.95、原奖励/clip、原一步 TD、网络尺寸、优化器、Replay 与更新节奏。当前 MADDPG 的一步 target 在末尾裁剪 [-10,10]；多步奖励求和后只裁一次通常不等价。优先做文档 A0 的严格一步对齐；C0 只诊断，C1 新目标语义后续单独审查，不把多步标签偷偷混入原 TD。

地图来源锁定后再做 RL 作用与分支一致性检查，避免地图和 Critic 同时变化导致归因不清。将来可做旧/新地图 x 原/新 RL 的 2x2 对照；地图对照标签与现有 B2/B3 方法身份始终区分。

**下一批建议只实现第 1 批的公共事件接口、默认关闭装配和必要定向检查；不启用新搜索决策，不进入后续训练。**

## 附：本轮阅读文件的 SHA256

以下哈希由本轮只读审计生成，仅绑定本说明的阅读依据，不是运行准入、源码演进审批或性能凭据。

| 阅读文件 | SHA256 |
| --- | --- |
| AGENTS.md | 72ef946a9c03f21527ff1b6db03eafe9cf41892ac66a7edee282b78a8d99c1a0 |
| docs/pdf/chapter3_map_research_design_20261017_v1.pdf | 6a37f3a332ba81e96a3d740f12c61ea6a155f26fcd89d9992703bdf1c6c6996f |
| docs/pdf/ch3_rl_124_20261018/第三章强化学习方案124_可编辑参考.md | 99fbbe6d9c9fa5804bcd373f9e079f2c3838e63695ee99c928c4536080847a2a |
| core/env/uav_env.py | 83b7ea143e8c203c4e0249c4c2f0b212eae05ca1b612941e7bc6245fa33d1556 |
| core/env/mission_env.py | 53ca90a07a7bf212db8615485f52a1a67071ab686e9588e8496282a8ca19176b |
| core/env/target_motion.py | 5a4459d15035740ca76899065b8e45ac64a99257eee6edf8db2f362dc41d478f |
| core/mapping/planning_state.py | 4b8d395c072fcc501bf0e6a98b5be8c6011757cd9455434f2a1c511c227103c9 |
| core/mapping/path_planner.py | 0415b51e1ac2c8f0e48b8b4effaf19b741adc6dca4a3203b2c2282654351c2aa |
| core/mapping/travel_cost_service.py | 8629dbf7f0e0ef59f03ecea4e865a644c6a012c1dca0ffe03b4c7bc66b09bbcb |
| chapter3_bser/candidate_generator.py | b8b46d413c64ef155a4b81a92cffaf783bc6b9fe5f56aabfb4b41393e4522ca8 |
| chapter3_bser/objective.py | eb705af260a8afc0cf9a2be3a522af2b9057f02f0adadcf19d86829db3ee9d7f |
| chapter3_bser/controllers/state_provider.py | 9e4034101d021edbc4b1853a0634669750cbf5bc6339a2e5287fcabe1afe9330 |
| chapter3_bser/controllers/path_tracker.py | 436970b1a83dd3d1da731692a1140fe2d68d8411ee49213b745cf4aa953d796e |
| chapter3_bser/d2/assembly.py | ab3beb6c1a56c56caceb5a4f1e319e09240148f94fda8f63d19a569c2e157dea |
| chapter3_bser/d2/allocator.py | 23392d7d81c3239946ec3be21f928643a7abdd4223913d761a38722409c7393e |
| chapter3_bser/d2/solver.py | cc08ab05aee6d38c6ffc94a0350cfe070cb16c144ffe18c7489a50546606c470 |
| chapter3_bser/d2/scored_controller.py | 9b0c05698aa6ea7195c225ee5cf597df5ef6a62df4dc246cb3813496eb1b59d5 |
| chapter3_bser/d2/planning.py | 10836ba019607ee6a21f0f35e4132eed7cacc19807d05cbb33386c8ed50ef08e |
| chapter3_bser/safety/allocators.py | ea3cf0ccf57154e5386f03fdaeece4c91e92d6c98222c78c680083b0a790c06e |
| chapter3_bser/safety/runtime.py | 95e0da8cde5528020f4f9f86ef52e0c56ee3715ddb0bfe961e9c3b51ac8fbb29 |
| chapter3_bser/safety/safe_guidance.py | 0c4e10808ce34e9268239aaff085cfad287d047dae9f18983384db0f1fd30edc |
| chapter3_bser/safety/planning_refresh.py | 9e47bb939590dd0f9c281910937c1ad7c6c7f0156a6b57234f9625312f212001 |
| chapter3_bser/integration/guided_env.py | 215c9335c8035395fe4a8da24ef2d31d02ec78e8d2444dba800f832928067a25 |
| chapter3_bser/integration/rmaddpg_bridge.py | 22ffbc0ee8f3e0228fe964c3b872c5b0b2190a9653d9358797a2a0617379d6e6 |
| chapter3_bser/online/controller.py | e43254a106079368e5bc6ea8186f47645ce0f68627b3b95fedeeace6c569fa67 |
| chapter3_bser/online/types.py | 623ddf947b87c49815aa95160b3928ff532d61056c38a42f6ef7022ade68e13d |
| chapter3_bser/learning/base_runtime.py | 32d2d79d4852ed62310ac8c5f612834d920d54923eb7c60399478c6c85a05a06 |
| chapter3_bser/learning/gated_runtime.py | 0ac0f0dc1adeaf31981ff9ab8a37d4d180695db8da089326ed017f6e49a48b4f |
| chapter3_bser/learning/config.py | 395a154c1fab1edf0e54c9fb29e737cb72a7c6bb51c2328a5c022141be480c24 |
| chapter3_bser/learning/provenance.py | 79371cca1466735542eb3d673c1b95aec851da9071cfb39fde97be921f3c55b0 |
| chapter3_bser/learning/reference.py | c0fa47dffa4ca516b402996a0191f1e1fe74eb960a9f331c091f87e9a6190f7d |
| configs/environment.json | d5881ea13b6f856e55f06051287761e9eca7c65739e03fb9232f6cadf5c1cb52 |
| configs/online.json | 9d49558d56662a58db7b5e601a42d0245828b0bd51c662cdedaa125da5e67d4d |
| configs/allocation.json | 9b28fc9101fd2ba127d403e6ef0045cf8727a1046b9cd1f1db29b09dacf5f30a |
| configs/b2.json | 41e46d4d9ae35f7808d3bb82ae859f83117580013f8c2e69d93f4b83203dce48 |
| configs/b3.json | 925b442da6a56d38e3c432ef5bb00d2e85ed43bd81a30dec48091806b2c64c09 |
