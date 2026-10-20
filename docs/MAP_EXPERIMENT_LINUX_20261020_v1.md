# 新旧地图各自适应后的 D2+RL 实验

本指南对应完整的原生地图训练协议：OLD_B2、NEW_B2、OLD_B3、NEW_B3 分别从新模型训练，在各自训练地图上评估；另有 OLD_ZERO、NEW_ZERO 两个零残差参考。不会加载或重标历史模型，不包含冻结策略跨地图迁移实验。所有代码在 Active 内，运行时不导入归档或其他仓库。

## 1. 实验与脚本

| 文件 | 用途 |
| --- | --- |
| scripts/linux_map_experiment.sh | Linux 分阶段启动器，缺少 --execute 只显示计划 |
| configs/map_comparison_v1.json | 首轮配置：2 个种子、300 回合、400 步、12 个固定验证任务、100 个最终测试任务 |
| configs/map_comparison_pilot_v1.json | 可选流程试跑：2 个种子、20 回合、400 步、50 个独立开发评估任务；首轮 300 ep 可直接跳过 |
| chapter3_bser/learning/map_suite.py | 准备配对任务、独立训练、原生评估、完成凭据与安全重入 |
| chapter3_bser/learning/map_contract.py | 地图身份、检查点约束、实际运行计时和地图机制指标 |
| chapter3_bser/learning/map_report.py | 六组汇总、逐种子结果、配对差值、交叉 bootstrap、验证曲线与收敛筛查 |
| tools/check_map_training.py | 8 个限步回合、4 个临时模型、每模型最多 4 轮更新，不恢复模型文件 |

当前默认为 **4 个 RL 组 × 2 个种子 × 300 ep = 8 个新模型、2400 个训练回合**，上限 960,000 个训练环境步。两个 ZERO 组不训练。独立固定验证集为 12 场，在 0/50/100/150/200/250/300 ep 共 7 个点离线评估：8 × 7 × 12 = 672 回合，上限 268,800 步。最终 8 个已训练策略和 2 个 ZERO 参考各测试 100 场，共 1000 回合，上限 400,000 步。全部评估预算明确独立于训练预算，实际常因提前终止而更少。

每个模型仍使用最终第 300 ep 的检查点比较，不按验证或测试分数挑选模型、不提前停止、不改变学习超参数。旧的 20 模型 × 1000 ep 配置已随来源/证据逐字节冻结在 provenance/baselines/map_experiment_20261020_v1/。

同一种子、同一种 B2/B3 的 OLD/NEW 任务共用：初始权重生成规则、冻结训练任务及顺序、每回合环境种子（training_seed + episode）、学习参数。模型、Replay、优化器和输出分别创建。脚本核对配对初始权重哈希，配置中除地图和输出目录外的差异会被拒绝。

OLD 在统一源码中走原默认关闭 D2；NEW 显式启用耦合搜索。两个地图共用原动力学、奖励、4 个智能体、28D/3D/124D、gamma=.95、网络、优化器和更新节奏。保留原单线程 CPU 学习，不引入 GPU 学习算法或参数变化。地图与学习方法分字段记录，NEW_ZERO 不重标为原始 D2。

## 2. Windows：检查并上传当前完整版本

当前已有 origin：https://github.com/SKT-T1otto/AUV-Thesis-Active.git。下面命令由用户手动执行；本次代码工作没有提交或推送。若你的 origin 已变更，以 git remote -v 为准，不重新初始化已有仓库。

```powershell
conda activate AUV
Set-Location E:\gym\code\WORKSPACE\AUV-Thesis-Active
$ErrorActionPreference = 'Stop'
function GitOK { & git @args; if ($LASTEXITCODE -ne 0) { throw 'Git 失败，停止。' } }
function PythonOK { & python @args; if ($LASTEXITCODE -ne 0) { throw 'Python 检查失败，停止。' } }

PythonOK -m tools.verify_source
PythonOK -m cli verify
GitOK status --short
GitOK remote -v
GitOK config core.autocrlf false
GitOK add -- .gitattributes .gitignore AGENTS.md PACKAGE_SHA256.json README.md cli.py project_paths.py requirements.txt core chapter3_bser configs checks tools scripts provenance verification docs
PythonOK -m tools.validate_checkout --index-only --execute
GitOK diff --cached --stat
GitOK commit -m 'Add paired native old-new map D2 RL experiments'
GitOK push origin HEAD
GitOK rev-parse HEAD
```

保存最后的完整 40 位提交 SHA。`.gitattributes` 保留原字节，不批量转行尾；Bash 启动器本身使用 LF。不要上传 work、outputs、模型文件或本地环境。版本归档和紧凑验证凭据属于来源证据，需要保留并上传。

## 3. Linux：克隆、依赖和当前版本验收

使用一个新目录。EXPECTED_COMMIT 必须替换为上一步打印的 SHA，不能保留占位符。以下操作不会启动正式训练。

```bash
set -euo pipefail
EXPECTED_COMMIT='粘贴Windows打印的完整40位SHA'
git clone --config core.autocrlf=false \
  https://github.com/SKT-T1otto/AUV-Thesis-Active.git AUV-Thesis-Active-map-exp
cd AUV-Thesis-Active-map-exp
git checkout --detach "$EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"

conda create -n AUV-Active-map python=3.10 -y
conda activate AUV-Active-map
python -m pip install -r requirements.txt
python -m pip check
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

python -m tools.verify_source
python -m cli verify
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CHECK="$PWD/work/verification/linux_map_$STAMP"
bash scripts/linux_map_experiment.sh verify \
  --output "$CHECK" --expected-commit "$EXPECTED_COMMIT" --execute
cat "$CHECK/validation.json"
VALIDATION="$CHECK/validation.json"
```

须看到最外层 passed=true、linux_execution=true、map_workflow_verified=true、正确 HEAD，以及当前完整测试库存无失败/跳过。当前库存为 130 项，具体以 configs/rebuild_check_contract.json 为准。

验收会用 4 个临时新模型做 4 个训练夹具回合和 4 个评估夹具回合，每回合最多 8 步、每模型最多 4 次完整多智能体更新。评估直接使用刚训练的内存模型，不恢复检查点文件。夹具的 batch/warmup=4/4 只用于机制验收；正式配置仍为 128/256。这不是性能测试。

当前本地验证来自 Windows，不能算作 Linux 通过。只有 Linux 上这次新生成的 validation.json 才用于后面的正式启动；训练启动还会核对源码身份、包哈希、Git HEAD 和干净检出。修改代码或上传新提交后，需要重新验收并准备新的 RUN。

本版证据保存在 verification/map_convergence_v1/：当前源码对应的 130 项检查全部通过；4 个临时模型共完成 8 个限步回合、36 个环境步，每模型 4 次完整更新，未恢复模型文件。暂停前完成的这批结果经源码哈希核对后保留，本次续做没有新增物理回合。完整 2 种子／300 ep 预算通过配置核验；实际生成器另用 7 个任务定义验证准备和隔离链路，未生成或执行完整训练任务。HTML 的 9 张图通过合成数据与 6 万条更新记录的脚本执行检查；合成数据不是学习结果。上述证据均不代表 300 ep 已收敛。

## 4. 可选：开发集小规模试跑

此阶段有实际训练，只有用户执行带 --execute 的 train 命令才开始。20 回合通常不足以证明学习收敛，主要用于检查流程、耗时、地图退化/回退和开发参数。开发评估 seed=827299，与正式测试 seed=92729 分开。

```bash
PILOT="$PWD/work/collision_terminal/map_pilot_$STAMP"
bash scripts/linux_map_experiment.sh prepare \
  --root "$PILOT" --spec configs/map_comparison_pilot_v1.json --execute

# 先查看计划和预算；这一行不执行训练。
bash scripts/linux_map_experiment.sh train --root "$PILOT" --validation "$VALIDATION"

# 明确执行后才创建新模型并训练。
bash scripts/linux_map_experiment.sh train --root "$PILOT" --validation "$VALIDATION" --execute
bash scripts/linux_map_experiment.sh evaluate --root "$PILOT" --validation "$VALIDATION" --execute
bash scripts/linux_map_experiment.sh summarize --root "$PILOT" --output "$PILOT/report" --execute
cat "$PILOT/report/report.md"
```

若实际训练未产生任何 Actor 更新，任务不会被标为“完成的学习实验”。查看 jobs/*/train/summary.json、failure.json 和日志；保留失败现场。根据开发结果调整研究配置时，将完整修改后的 spec 放到新的 work 输入文件，冻结新来源/协议要求后另建 RUN；不要修改已准备任务的 inputs 或 plan。

## 5. 首轮实验：8 个模型训练、收敛诊断、最终测试

```bash
RUN="$PWD/work/collision_terminal/map_main_$STAMP"

# 只生成训练、验证、测试场景和配置，不创建模型、不训练。
# 做过 pilot 时显式排除其任务；没有做则保持空参数。
EXCLUDE_ARGS=()
if [[ -n "${PILOT:-}" ]]; then EXCLUDE_ARGS+=(--exclude-run "$PILOT"); fi
bash scripts/linux_map_experiment.sh prepare \
  --root "$RUN" --spec configs/map_comparison_v1.json \
  "${EXCLUDE_ARGS[@]}" --execute

cat "$RUN/plan.json"

# 再次打印训练请求，不执行。
bash scripts/linux_map_experiment.sh train --root "$RUN" --validation "$VALIDATION"

# 正式训练：4 个学习组 x 2 个种子 x 300 ep，模型和 Replay 都独立。
bash scripts/linux_map_experiment.sh train --root "$RUN" --validation "$VALIDATION" --execute

# 收敛诊断：同一组 12 个验证任务，关闭探索，检查 0/50/.../300 ep。
# 这里读取本次新生成的检查点，不续训、不改变模型，不读最终测试集。
bash scripts/linux_map_experiment.sh diagnose --root "$RUN" --validation "$VALIDATION" --execute

# 各任务的收敛筛查结果和 HTML 曲线。
cat "$RUN/jobs/NEW_B2_seed2729/diagnostics/convergence.json"
# 将 jobs/<job>/diagnostics/learning_curves.html 下载到本机浏览器打开。

# 最终测试：只在各模型自己的训练地图上评估；同时评估两个 ZERO 参考。
bash scripts/linux_map_experiment.sh evaluate --root "$RUN" --validation "$VALIDATION" --execute

# 六组任务全部完成并通过来源/策略/地图/场景核验后，才生成完整汇总。
bash scripts/linux_map_experiment.sh summarize --root "$RUN" --output "$RUN/report" --execute
cat "$RUN/report/report.md"
```

prepare 会生成并冻结有序训练场景、共同固定验证场景、共同测试场景、原始生成清单和每任务配置。生成器原本可能给不同种子的场景同名；本脚本明确增加命名空间并保留 source_scenario_id，不改物理数据。隔离检查仍同时核对 ID、seed 和物理指纹，不能靠改名绕过。--exclude-run 可多次使用，会逐字节保留开发任务清单并拒绝正式测试与这些任务重叠。

用户若希望明确一次执行训练和评估，可使用 run --stage all 替代上面的 train/diagnose/evaluate 三步（启用 convergence 时）；仍须显式 --execute 和 --validation。summarize 单独执行。

只运行部分组或种子时：

```bash
bash scripts/linux_map_experiment.sh train \
  --root "$RUN" --groups OLD_B2 NEW_B2 --seeds 2729 \
  --validation "$VALIDATION" --execute
```

同一 RUN 有互斥锁，按序运行任务。再次执行会核验并跳过有完成凭据的任务；不重复评估已完成策略。不支持恢复未完成训练。已有非空输出但没有完成凭据时拒绝覆盖，保留现场另建 RUN。不要手动删除运行锁；确认原进程状态后再处理。tmux 中需重新激活环境，保留同一个 RUN 和 VALIDATION。

## 6. 输出与结论范围

每个学习任务保存 config、episodes、training_metrics、完整回合检查点、最终检查点、summary 与 training_receipt。检查点绑定训练地图、完整研究配置摘要、源码、初始权重摘要、训练任务和学习身份。评估从检查点恢复后自动使用其原生地图，不提供静默切换评估地图的选项。

report/ 包含：

- groups.csv：六组的 Found、Success、碰撞、失败惩罚发现步数、折扣回报、运行耗时及回退；学习组另报告种子标准差。
- per_seed.csv：每个最终策略的原始汇总和权重摘要。
- paired_deltas.csv：NEW-OLD 配对差值与 95% 交叉 bootstrap 区间；学习种子和共同测试场景分别重采样。单种子区间仅含场景不确定性，ZERO 没有学习重复。
- learning_curves.csv：逐回合训练回报、Found、Success、碰撞和累计环境步。
- validation_curves.csv：固定验证集上的全部检查点曲线；convergence.json：逐种子的判断。
- jobs/<job>/diagnostics/learning_curves.html：独立可打开的图表，包含验证指标、20 回合滑动训练回报、100 次更新滑动 Actor/Critic loss；鼠标悬停读取值。原始数据保存在 validation_curve.csv、optimizer_curve.csv 中。
- summary.json / report.md：完整性、协议、来源和可读结果。

主要系统比较是 NEW_B2-OLD_B2、NEW_B3-OLD_B3；同时用 NEW_ZERO-OLD_ZERO 观察无学习时的地图贡献。Found、Success、回报通常希望上升，碰撞、失败惩罚发现时间和耗时通常希望下降，不能只比较成功回合时间。成本和回退同样需要报告。成功提案的规划计时/查询数不等于全流程成本；runtime_step_seconds 包含环境、Actor、规划与回退。

groups.csv 的 runtime_seconds 是各物理步执行时间之和，**不含回合初始化、学习更新或模型文件恢复**。成功提案的 core_planning_seconds 包括初始提案；训练 summary.json 的 wall_seconds 包括初始化和学习。若报告完整进程资源，请在首次实际执行时用 Linux 的 /usr/bin/time -v 包住 train/evaluate，并用新的日志文件保留结果，不能将步骤计时当成完整系统墙钟。例如：

```bash
RESOURCE_TAG="$(date +%Y%m%d_%H%M%S)_$$"
/usr/bin/time -v -o "$RUN/evaluation_resources_$RESOURCE_TAG.txt" \
  bash scripts/linux_map_experiment.sh evaluate \
  --root "$RUN" --validation "$VALIDATION" --execute
```

该命令应替换首次 evaluate 命令。已经完成的评估会被核验后跳过，重入耗时不代表评估成本；需要每组独立资源记录时，在首次评估时配合 --groups/--seeds 分组启动。

此版本交付实验执行能力，不承诺地图或 RL 一定提升性能，不转移历史 E0/正式性能验收。冻结策略跨地图迁移、PDF 的全部地图消融和后续 RL 算法方案不属于这套原生训练主实验。

## 7. 如何判断 300 ep 是否够用

先运行 train，再运行 diagnose；诊断不需要最终测试结果。0 ep 为按该种子重新构建并核对初始权重哈希的未训练 RL 策略，与 ZERO 参考不同。训练仍带原探索噪声，验证一律 explore=False。每次验证使用完全相同的 12 个物理场景和场景创新种子，OLD/NEW、B2/B3 及两个训练种子共用；验证、训练和最终测试之间按 ID、场景种子和物理指纹隔离。诊断只离线读取本次新检查点，不向 Replay 添加数据，不更新优化器，并保留调用前 RNG 状态。

诊断规则在配置里预先冻结，逐种子使用最后三个点（200/250/300 ep）：

- Success、Found、碰撞率各自的最大值减最小值均不超过 0.10（10 个百分点）。
- 失败按 400 步惩罚的发现步数均值，其极差不超过 0.05 × 400 = 20 步。
- 折扣回报的极差不超过 0.05 × max(|0 ep 验证回报|, 1)。
- 同时要求有真实 Actor 更新、有限 loss；后三点平均 Success 或 Found 相较 0 ep 至少改善 1/12，或者惩罚发现步数至少减少 20 步；平均碰撞率不得比 0 ep 增加超过 0.10。

状态含义：provisional_plateau 表示“有任务改善且尾部暂时趋稳”；still_changing 表示“300 ep 内仍变化”；stable_without_clear_task_gain 表示“稳定但未显示有效任务改善”；stable_with_collision_regression 表示“稳定且有其他改善，但碰撞退化”；no_learning_updates 表示缺少有效更新或 loss 异常。最后三点前需要初始点，不足时为 insufficient_checkpoints。

**这是一套探索性的工程筛查，不是收敛证明。** 12 场的单种子率分辨率约为 8.3 个百分点，噪声较大；应同时看两条种子曲线和完整指标，不能仅凭平滑回报、loss 下降或群组平均作结论。两个种子均满足筛查时报告 all_seeds_provisional_plateau；任一种子仍明显改善或振荡，就不能说该组已趋稳。原始未平滑验证点、种子差异、碰撞和地图回退应一起报告。需要更强证据时另设更大验证集或更长训练，预算与停止规则另行冻结。

可先执行一对配对模型，完成该对的诊断，再继续其它模型，无需先训练全部 8 个：

```bash
bash scripts/linux_map_experiment.sh train --root "$RUN" \
  --groups OLD_B2 NEW_B2 --seeds 2729 --validation "$VALIDATION" --execute
bash scripts/linux_map_experiment.sh diagnose --root "$RUN" \
  --groups OLD_B2 NEW_B2 --seeds 2729 --validation "$VALIDATION" --execute
```

不要根据最终 100 场测试挑选检查点。若看过本轮最终测试后决定调整超参数或训练预算，这批测试结果应作为探索结果；后续确认实验需要新生成独立测试集，并通过 --exclude-run 排除本轮任务。当前训练不支持断点续训；300→更长预算需新 RUN 从头训练，本轮模型和输出全部保留。
