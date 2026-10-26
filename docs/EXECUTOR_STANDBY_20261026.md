# 搜索阶段执行器待命安全修补 v1

这是明确独立标识的导航行为修补，不是声称结果完全相同的计算加速。默认未配置或 `enabled=false` 时仍构造原 `SafetyBridge`。原 D2、环境动力学、碰撞终止、奖励、发现与交接、B2/B3 网络及学习过程、W64 地图推断内核均保持原文件字节。

## 已有 Linux 结果能确定什么

用户提供的 `map_diagnostics_pilot_20261026_004516_236696` ZERO 结果中：

| 地图条件 | 执行器 3 在 Search 阶段碰撞的场景后缀 | 碰撞步数 |
|---|---|---|
| OLD_ZERO | 0002、0006 | 96、82 |
| NEW_ZERO | 0002、0006、0011、0012 | 55、30、29、27 |

Search 是任务阶段；此时执行器仍会向待命点移动。因此可以确认“搜索期间执行器移动发生物理碰撞，导致整场失败”，原地图也存在这种情况。仅凭这些汇总不能确定是候选点错误、沿途路径失效、转弯惯性、未观测障碍或其组合。

另有 NEW_ZERO 的 `missing_original_standby` 两条记录，分别来自 0002、0011；这表示该次提案没有兼容待命候选，触发回退，不等于执行器已经碰撞，也不能与四场执行器碰撞一一对应。其他 81 条回退原因为 `missing_public_safe_route`。这批记录中未见计算预算超时；场景运行到 400 步的 `timeout` 是任务时限。

## 修补范围与边界

原 `SafetyBridge` 在每步安全复检中显式跳过执行器。新增 `ExecutorStandbyBridge` 复用原搜索器安全处理，只在目标未发现、任务未完成时检查执行器：

1. 用当前公开占据体素和原 clearance 检查当前点到跟踪点、剩余路径。不会读取真实障碍布局来作决策。
2. 用公开执行器速度、加速度、阻力、浮力和执行器先验增益，预测 8 步名义跟踪，再预测 16 步随当前位置更新的制动。单条路线预测最多 24 步；含原路线、替代路线和纯制动检查，每次编译最多 64 个名义步。不扩大地图内核、粒子数或历史窗口。
3. 路径或预测被拒绝后，每次编译最多发起一次原公开图 A* 查询；替代路线必须通过几何与预测复检。成功后保留临时路线，待命任务及最终目标不变。分配签名变化立即失效；公开几何每步复检。
4. 无可验证路线时发出原动力学下的 hold 导航意图，并通过原 repair-v2 hold gate 抑制学习残差。分别记录 `nominal_braking_checked` 与 `unverified_braking`，不会把速度直接清零、修改动作映射或吞掉碰撞终止。
5. 同时更新 `agent_assignments[3]` 和 `executor_assignment`。发现目标后清除临时路线，继续原交接和执行控制。

名义预测未包含未知流场及学习残差，有限 16 步制动也不等于证明完全停止。公开占据地图尚未看到的障碍、当前位置已进入膨胀占据体素、制动距离不足仍可能失败；日志保留这些情况。它是有界计算的保护，不提供无碰撞保证。桥接计时包括原搜索器处理与新增执行器处理，不能当作新增保护的独立耗时。

## 新实验

`configs/executor_standby_pilot_v1.json` 与 `cli standby-experiment` 生成四组独立身份：

| 目录 | 地图 | 执行器保护 | 学习 |
|---|---|---|---|
| guard_off / OLD_ZERO | 原地图 | 关闭 | 无 |
| guard_on / OLD_ZERO | 原地图 | 开启 | 无 |
| guard_off / NEW_ZERO | bounded v3 W64 | 关闭 | 无 |
| guard_on / NEW_ZERO | bounded v3 W64 | 开启 | 无 |

四组使用同一份冻结的 12 个开发场景、相同场景顺序及创新随机种子，每场最多 400 步，共 48 次评估、零训练作业。生成种子仍为 827699，以直接检查上述问题场景；这些数据已用于诊断，不能再称为独立留出验收集。`train_episodes=30` 是沿用计划格式的字段，ZERO 不创建训练作业。

依次验证：执行器 Search 碰撞是否减少；是否产生长期 hold 或不可验证制动；搜索发现率、发现步数、任务成功率是否退化；发现后是否仍长期无法完成；公开路径查询和整步运行时间是否增长。报告同时保留搜索器碰撞及地图提案回退，避免把它们归因到执行器。

`executor_collisions.csv` 每场保留碰撞对象/阶段、保护检查/重规划/制动次数及最后一次搜索期保护原因。完整 64 条尾部轨迹在原生 `episodes.jsonl` 的 `planner.executor_standby_guard.trace`。`guard_paired_deltas.csv` 给出每种地图开启减关闭的配对差值；每组原生地图报告也保留 found/success/collision、参与率、回退和耗时。

先完成 ZERO 再决定是否做 B2/B3 新身份训练。此轮不自动载入或复用旧模型，也不根据 32 步机制检查宣称长程修复或性能提升。

## 本地验证结果

当前源身份 `f7f1ae9271cc744de6e769fbb705d3227b06f1e6935bba51bbc5812a76440d97` 通过 238 项回归，无失败或跳过；四个新增入口的非执行模式均未创建运行目录。

| 新地图场景 | 关闭保护 | 开启保护 | 开启组诊断 |
|---|---|---|---|
| 0006 | 第 30 步执行器碰撞 | 32 步截断前无碰撞 | 第 12 步开始拒绝原待命路径；9 次名义制动检查、1 次路线修复 |
| 0011 | 第 29 步执行器碰撞 | 32 步截断前无碰撞 | 第 14 步开始拒绝原待命路径；6 次名义制动检查、1 次路线修复 |

这些路径被拒绝时，原待命分配仍标记为 `reachable=true`；修补后的公开图查询曾返回 `no_start_connector`，随后在第 20 步取得可验证替代路线。这直接展示“旧待命指令仍可达，但最新公开几何已不接受其路径”的缺口。它没有证明所有历史碰撞同因，也没有证明两场能完成 400 步任务。

本轮严格计入 8 次物理初始化：前一候选版完成 2 次原地图关闭保护检查，第 3 次初始化暴露实时安全视图不含执行器编号；改用规划状态中的身份后，当前源完成余下 5 次检查。共完成 7 场、219 个物理步、零学习模型、零 checkpoint 文件恢复。前两场保留原源身份，附精确的关闭保护源绑定；失败初始化原记录未覆盖。原地图 0011 的开关两组均在 32 步截断前无碰撞；原地图 0006 的开启组本地没有重跑，留给 Linux 的完整矩阵。全部证据位于 `verification/executor_standby_v1/<源身份>/`。

## Linux 执行

在既有仓库与 AUV 环境内更新，不必重新克隆或重建环境。将 `REV` 设为本次交付的完整提交号：

```bash
conda activate AUV
git -c http.version=HTTP/1.1 fetch origin main
git checkout --detach "$REV"
git rev-parse HEAD

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CHECK_STANDBY="$PWD/work/standby_checkout_$STAMP"
RUN_STANDBY="$PWD/work/collision_terminal/standby_pilot_$STAMP"
REPORT_STANDBY="$PWD/work/standby_report_$STAMP"

python -m tools.validate_checkout --output "$CHECK_STANDBY" \
  --require-linux --expected-commit "$REV" --execute &&
python -m cli standby-experiment prepare --root "$RUN_STANDBY" --execute &&
python -m cli standby-experiment run --root "$RUN_STANDBY" \
  --validation "$CHECK_STANDBY/validation.json" --execute &&
python -m cli standby-experiment report --root "$RUN_STANDBY" \
  --output "$REPORT_STANDBY" --execute
```

Linux 验证会运行全部回归和 8 次不超过 32 步的 ZERO 机制检查，无训练、无 checkpoint 文件恢复。新的机制检查使用既有场景 0006、0011 的短程截断，在两种地图和两个保护状态下对照。只有完整 400 步评估才能回答较晚碰撞、长期等待及执行完成问题。Git 网络失败时停在更新步骤，不能继续把旧代码标记为新版本。
