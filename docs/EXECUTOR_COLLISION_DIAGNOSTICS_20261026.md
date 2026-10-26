# 老地图搜索期执行器碰撞：原生诊断与配对复现

本轮只增加诊断工具、测试和来源准入元数据，不修改原D2、v1/v2保护、B2/B3 repair-v2、地图推断、物理、奖励或学习配置。先用OLD地图（research_config=None），不启动新地图、v2矩阵或训练。

完整Linux旧实验属于 `aef6833a6b80dcede92561b6497aafc1d5ad60e7`；本轮父版本是 `a5ab10de1ede2fbaf7802186ead5900ccd4ec266`。更新后的诊断器比较当前原版分支与历史原生记录，不将旧结果改为新源码结果。父清单、旧验证证据与旧权重身份均保留。

## 本轮要回答的问题

默认只运行三组已明确识别的问题事件，每组“无记录/有记录”两遍：

| case | 冻结条件 | 原碰撞事件 |
|---|---|---|
| baseline_0002 | guard_off / OLD_ZERO | 3号执行器，Search，第96步，障碍0顶面 |
| baseline_0006 | guard_off / OLD_ZERO | 3号执行器，Search，第82步，障碍1侧面 |
| guard_v1_0011 | guard_on / OLD_ZERO | 3号执行器，Search，第101步；此前记录unverified_braking/invalid_start |

每遍沿用冻结配置的400步上限，零训练、零权重加载。若原事件复现，应在96/82/101步提前终止；最多6遍×400步是**手动诊断研究**，不是自动机制检查。可先通过 `--cases baseline_0002 baseline_0006` 只做原版两场、共4遍。

原版两场目前缺原始待命目标和地图，不能断定是目标余量不足。v1的0011碰撞前距语义目标8.107米、距障碍约5.10毫米，证明该例发生在远离最终目标的制动回退过程中，但没有解释更早为何进入该位置。

## 记录器如何保持原行为

入口为 `python -m cli executor-diagnose prepare|run|report`，对应 tools/ 下的独立工具，生产环境和策略不导入这些工具。

1. 每步复制已安装指导、原始语义分配、完整路径/剩余路径/索引、4个智能体的物理状态、实时OLD占据/未知/自由/概率/信念数组、缓存规划状态和完整图。记录provider最近刷新时刻、原因与计数。
2. 旁观原 `_apply_agent_dynamics` 调用，在**实际积分前**记录环境最终使用的导航目标、速度及原始动作，在调用后记录原先验、实际残差、实际加速度、物理碰撞事件。原方法接收同一组参数，仅调用一次；记录器退出时恢复原方法查找状态。
3. 旁观原本就会调用的 `observe_controller_result` 和 `compile_guidance`，复制已有事件/可达性暂缓/接受拒绝原因，不再运行事件检测、求路或指导编译。终止步没有新的控制器结果时明确为null，不拿上一物理步的结果冒充终止步结果。
4. 不调用 provider刷新、planning_views.live、TravelCostService.query、图构建或tracking_target，不添加随机采样。每次步前采集均比较采集前后状态、RNG、规划缓存和指导签名。
5. 数据按内容哈希去重，原始记录由完成收据绑定。报告再次校验文件哈希、时钟/签名数量、步序及碰撞事件与原生摘要的一致性。篡改日志或把比较标记改成true都会被拒绝。

“无记录/有记录”的逐步签名比较包括位置、速度、实际控制相关缓存、原地图、缓存图、全部指导、跟踪索引、RNG、动作、奖励、观测和执行掩码。v1摘要只排除两个墙钟计时字段 `compile_seconds`、`max_compile_seconds`，其事件计数和结果都要一致。记录开销不能当成原算法运行时间。

原始场景真实AABB不供新增观察器控制决策使用；只在环境关闭、记录完成收据写出之后，由独立report阶段读取，用于几何诊断。原环境自身正常碰撞判定和原28D观测均保持不变。

## 报告如何区分原因

每场保留以下相互独立的证据，不用单个阈值自动宣布主因：

- **目标与驻留**：原始语义待命目标距离、目标对本次碰撞障碍的余量、hold状态和先前轨迹；实际tracking点单独保存，不替代语义目标。
- **路径与跟踪**：原始完整路径、原始剩余路径、当前到tracking连接段、实际运动段，各自与真实AABB的交点/最小余量；位置到原始折线的偏离距离不通过预先插入当前位置人为变成零。
- **地图与更新**：当时实时地图和缓存地图对同一连接段的已占据/未知状态、规划图revision、provider刷新和原控制器的有效性暂缓结果。
- **采样与连续几何**：复制原OLD采样公式进行事后计算，与独立闭体素求交oracle对照。两者可能在角点接触上不同；该差异并不自动证明历史碰撞由采样引起。
- **惯性、制动与扰动**：积分前速度、实际先验/残差/加速度、候选速度及公开参数下的先验重建。碰撞后速度归零不冒充碰撞前速度。仅在无速度/边界裁剪时估计水平扰动，明确包含数值误差，不读取隐藏流场真值。
- **离线制动反事实**：对碰撞前最后16步，以固定当前位置锚点、冻结推断扰动，做最多16步的纯数值制动预测，事后与真实AABB求交。它不是环境重放、不是新的控制算法，也不证明当时真实流场保持不变；只能作为解释性证据。

离线制动近似不复现墙面裁剪和碰撞后的反馈，只分析候选制动轨迹的首次障碍接触可能性。冻结场景小数与环境内部float32转换可能有末位差异；精确复现以原生collision_records逐字段比较为准，不把独立几何计算的微小浮点差异解释为新机制。

报告先确认记录器全程行为一致，再比较历史原生终端事件的actual_length、发现步、成功、终止原因、碰撞步/阶段/智能体和完整碰撞记录。完全一致仅称“终端事件精确匹配”；历史没有完整轨迹，不能声称历史全轨迹也相同。复现或记录一致性失败时仍保留数据，但不能用该数据确认历史主因。

## Linux更新与当前验证

本地发布验证（2026-10-27完成）：源身份 `d1a89107620893f541e011bfd19ea85f151310a66a18f1d4980764d6aacba4a5`，288项回归全部通过，无失败/错误/跳过；6次OLD配对前缀初始化共192物理步，三组逐步签名和行为摘要均一致；6个入口的非执行模式均未创建输出。另有零环境初始化的合成碰撞帧检查，覆盖完整事后报告链，不是历史碰撞复现。

当前证据位于 [verification/executor_diagnostics_v1](../verification/executor_diagnostics_v1/d1a89107620893f541e011bfd19ea85f151310a66a18f1d4980764d6aacba4a5/receipt.json)。实际先验的独立重建误差在数值容差内，积分前导航点与已安装tracking点一致，执行器学习残差均为零。没有把这些32步结果当成第82/96/101步事件结果。

继续使用已有目录及AUV环境。把REV设为本次交付的完整提交号。下列验证会运行全部回归及**6次、每次最多32步的OLD配对前缀检查**：原版0002/0006及v1的0011各无记录/有记录两遍，无模型训练或检查点恢复。

```bash
cd ~/AUV-Thesis/AUV-Thesis-Active/AUV-bounded-v3
conda activate AUV
REV="本次交付的完整提交号"
git -c http.version=HTTP/1.1 fetch origin main &&
git checkout --detach "$REV" &&
git rev-parse HEAD

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
DIAG_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
DIAG_CHECK="$PWD/work/executor_diagnostic_checkout_$DIAG_STAMP"
DIAG_RUN="$PWD/work/collision_terminal/executor_diagnostic_$DIAG_STAMP"
DIAG_REPORT="$PWD/work/collision_terminal/executor_diagnostic_report_$DIAG_STAMP"
DIAG_HISTORY="$PWD/work/collision_terminal/standby_pilot_20261026_164043_236696"

python -m tools.validate_checkout --output "$DIAG_CHECK" \
  --require-linux --expected-commit "$REV" --execute &&
python -m cli executor-diagnose prepare --root "$DIAG_RUN" \
  --experiment "$DIAG_HISTORY" --execute &&
python -m cli executor-diagnose run --root "$DIAG_RUN" \
  --validation "$DIAG_CHECK/validation.json" --execute &&
python -m cli executor-diagnose report --root "$DIAG_RUN" \
  --output "$DIAG_REPORT" --execute
```

更新失败时停下，不能用旧源码贴上新版本号。Linux验证文件必须匹配当前源码、完整提交和PACKAGE，旧aef6833的验证文件不适用于更新后的正式诊断入口。已有历史实验本身只读，prepare将指定的原生manifest/config/receipt/episodes复制到新的诊断输入目录并验证绑定。

若先只重放原版两场，在prepare命令追加 `--cases baseline_0002 baseline_0006`。先确认有/无记录的行为一致及历史事件匹配，再决定是否运行v1的0011，使用新目录，不能覆盖。

所有入口缺省为非执行模式，不创建环境或运行目录。run必须实际处于Linux、具有当前重建准入和当前Linux验证，并显式添加--execute。无法绕过这些条件在Windows自动运行82/96步长程实验。

## 贴回哪些结果

保存完整DIAG_RUN和DIAG_REPORT。每场原始数据在 `jobs/<case>/recorded/trace.jsonl`、`blobs/`，纯净对照在 `plain/`，本次运行的版本与比较在 `run.json`。

报告目录每场有 `analysis.json`、`executor_timeline.csv`，总报告为 `summary.json`，收据绑定报告文件。可先贴回以下摘要，再按需要读取完整尾部轨迹：

```bash
python - "$DIAG_RUN" "$DIAG_REPORT" <<'PY'
import json, sys
from pathlib import Path
run=json.loads((Path(sys.argv[1])/"run.json").read_text())
report=json.loads((Path(sys.argv[2])/"summary.json").read_text())
print(json.dumps({k:run[k] for k in ("source","episodes","recording_parity_passed","all_original_terminal_events_exact","cases")},ensure_ascii=False,indent=2))
for a in report["analyses"]:
    x=a.get("collision_frame",{})
    print(json.dumps({"case":a["name"],"executor_collision":a["executor_collision"],
      "step":a.get("collision_step"),"phase":a.get("collision_phase"),
      "motion_audit":a["motion_audit"],"hypotheses":a.get("hypotheses"),
      "last_frame":x,"braking_counterfactuals":a.get("braking_counterfactuals")},ensure_ascii=False,indent=2))
PY
```

## 验证边界

来源校验要求所有core和原chapter3运行模块（除来源元数据）与a5ab10d逐字节一致，原来源准入函数在只去除新诊断元数据后AST一致。a5ab10d的源码/文档作为数据归档，已发布验证和嵌套基线按哈希不可变引用，不导入归档代码。

本轮自动环境检查只覆盖32步前缀，不能证明历史82/96/101步碰撞已复现，也不证明v2保护有效。最终v2已有252项回归与早期候选243步的证据边界继续保留；本次新增前缀检查不能重新标记为v2控制验证。

本轮不修改或上传Windows用户目录3090结果，不恢复旧权重，不启动B2/B3训练。诊断证据齐备后再确定控制/RL修改方案。
