# 地图诊断与下一轮实验

本轮为 `coupled_search_bounded_v3` 增加观察与报告，不改变推断、路径评分、事件触发、奖励、学习器和物理过程。M=3、每图N=32、H=16、W=64、3秒推断和20秒规划预算保留。观察代码本身有少量计时/记录开销，临近墙钟阈值的执行不保证逐步轨迹等价。

## 新增记录

- 推断记录扫描与地图记忆、几何修复、目标更新、恢复种子、窗口重放；v2另记录几何重建、先验和全历史重放。每个尝试均记录成功/失败、阶段、首个异常和最终错误。恢复成功也保留最初耗尽原因。
- 规划记录公开状态、目标预测、路径生成、路径预测与签名、压缩、联合选择及执行器待命选择。失败提案和评分尝试也进入报告；评分中的D2回退耗时单独列为阶段。
- 回退记录全部原因及每个决策步的首个原因，区分 `core_failed_this_step` 连带症状。正常完成和异常回合都保存诊断。原健康检查不放宽。
- 提案被控制器接受后才建立路径来源。实际安全桥安装后区分新模块原路径、D2原路径、被修改路径、保持和来源未知。用下一条公开物理转移累计实际使用的搜索器步数，包含终止步；发现目标后另计执行阶段。
- 原始轨迹仍可能受策略动作和动力学影响。“新模块原路径占比”表示安装指引下的执行覆盖，不表示动作完全由地图产生，也不证明任务性能的因果改善。
- 尾部记录容量1024；总计数保留全回合，记录截断显式标识。OLD组或旧输出缺失诊断时保留空值，不填成0。

报告增加 `participation.csv`、`compute_attempts.csv`、`fallback_reasons.csv`（存在回退时）和 `diagnostics.md`。原 Found、Success、碰撞、惩罚发现步数和原生检查点验证继续保留。部分组可单独汇总，明确标识未覆盖全部计划组。

## Linux准备

同步聊天中给出的本次提交，复用现有AUV环境；旧RUN和旧权重保留。先运行以下准备，当前源码验证通过后再执行后续阶段。

```bash
REV='替换为聊天中本次发布的完整40位SHA'
git -c http.version=HTTP/1.1 fetch origin && git checkout --detach "$REV"
conda activate AUV
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CHECK_DIAG="$PWD/work/verification/map_diagnostics_linux_$STAMP"
python -m tools.validate_checkout --output "$CHECK_DIAG" --expected-commit "$REV" --require-linux --execute
```

## 第一阶段：同输入受控长流，不训练

```bash
PROBE="$PWD/work/map_diagnostic_probe_$STAMP"
python -m tools.check_map_diagnostics --output "$PROBE" --execute
cat "$PROBE/report.md"
```

默认384批、每批104条扫描射线。所有条件使用同一份保存并校验哈希的公共输入流：

| 条件 | 验证目的 |
|---|---|
| W32、W64、W128自然流 | 状态有界、普通推断耗时与长程稳定性 |
| 三种窗口在192/320步人工目标耗尽 | 边界滑动后能否恢复、恢复耗时、每次重放≤W、两次总工作≤2W |
| 三种窗口人工几何耗尽 | 几何迁移零质量分支是否可诊断并恢复 |
| 无边界提议消融 | 恢复分支下的机制差异；不据此推断真实恢复频率 |
| 关闭恢复 | 第一次人工耗尽必须明确失败，属于预期负控 |
| 窗口地图自然流 | 重组成本及有限记忆行为 |

该流的扫描探针和检测轨迹独立指定，不是真实物理闭环；不创建模拟环境、模型或加载检查点。它验证机制和同输入计算，不能比较真实任务成功率。`passed`表示预期检查满足，预期失败负控不会伪装成完整运行。保留所有失败的尝试与诊断。

## 第二阶段：Linux原地图/新地图零学习对照

完成上面的Linux准备后，创建新RUN。源验证自带的接入检查最多8个短回合，不是正式训练。

```bash
RUN_DIAG="$PWD/work/collision_terminal/map_diagnostics_pilot_$STAMP"
python -m cli map-suite prepare --root "$RUN_DIAG" \
  --spec configs/map_diagnostics_pilot_v1.json --execute

python -m cli map-suite run --root "$RUN_DIAG" --stage evaluate \
  --groups OLD_ZERO NEW_ZERO --validation "$CHECK_DIAG/validation.json" --execute

ZERO_REPORT="$PWD/work/map_diagnostics_zero_report_$STAMP"
python -m cli map-suite summarize --root "$RUN_DIAG" \
  --groups OLD_ZERO NEW_ZERO --output "$ZERO_REPORT" --execute
```

两组各12个相同开发场景、每回合最多400步，共24次评估，没有RL训练。OLD_ZERO是原D2地图，NEW_ZERO是新版长期地图加零学习残差。检查：搜索前/后失败位置、搜索评分覆盖、新模块原路径实际执行比例、保持和安全修改、回退首因、失败计算耗时。搜索在发现后正常结束，不追求占整个回合的比例。

旧v2的全历史超时已保留为失败对照，不在本轮强制重跑。超时或健康失败时保留 `failure.json` 和 `map_failure_*.json`，先查看首个原因、最终阶段与重放工作；不覆盖目录、不抬预算、不绕过健康检查重启。

## 第三阶段：按需启动B2，再启动B3

ZERO诊断定位完成后再手动执行：

```bash
python -m cli map-suite run --root "$RUN_DIAG" --groups OLD_B2 NEW_B2 \
  --validation "$CHECK_DIAG/validation.json" --execute

B2_REPORT="$PWD/work/map_diagnostics_b2_report_$(date +%Y%m%d_%H%M%S)_$$"
python -m cli map-suite summarize --root "$RUN_DIAG" --groups OLD_B2 NEW_B2 \
  --output "$B2_REPORT" --execute
```

每组使用种子2729、独立从头训练30回合、每回合≤400步；第0/10/20/30回合各评估6个固定验证场景，最终各评估12个开发场景。两组共60个训练回合、48个验证回合和24个最终评估回合，不复用旧模型。B3使用同样设置，将组名替换为OLD_B3/NEW_B3并使用新的报告目录。

本配置为诊断pilot，不是正式性能验收。平台期只表示小验证集结果稳定；需核查更新数、检查点策略差异和实际任务质量。多种子独立最终测试留在机制与参与问题解决之后；开发场景不可冒充最终测试集。

## 本地工程验证

最终源码身份 `5fb615b191b137fd4700ae08d45896515f8e1c037c3e53d04fa6a9f79eb729c6` 的228项完整回归通过，无失败或跳过；独立来源验证确认移除观察语句后的算法AST一致，原D2、环境、奖励、学习器和几何内核保持原字节，严格健康条件保留。

同输入384批测试的12项条件均满足预期，关闭恢复在第192步人工耗尽处按预期失败。三种窗口的目标/几何耗尽条件均恢复2次，每次最多重放各自W步。原始输入和尝试记录压缩保存，保留未压缩公共流哈希。这不是Linux闭环任务成绩。

本轮两次观察版本接入共8个短回合、256个物理步、4个临时模型，每模型4次完整更新，评估直接使用内存模型且策略保持不变，无检查点文件恢复。最后一次短接入及长流测试之后，仅修复ZERO没有训练目录时诊断日志不落盘的异常分支，并更新原失败用例覆盖该分支；没有继续创建物理回合。最终回归覆盖该修改。

证据中的 `functional_source_binding.json` 校验短接入记录保存的全部150个生产文件哈希：仅 `map_contract.py` 的失败日志目录创建语句有差异，其余生产文件字节和成功运行函数AST一致。短回合、探针及检查点身份保留原记录，没有重标为最终身份。最终Linux验证仍必须在用户平台运行。

证据位置为 `verification/map_diagnostics_v1/5fb615b191b137fd4700ae08d45896515f8e1c037c3e53d04fa6a9f79eb729c6/`。新父版归档约3.30MiB，旧ZIP只保留不可变哈希引用，没有再次嵌套大归档。
