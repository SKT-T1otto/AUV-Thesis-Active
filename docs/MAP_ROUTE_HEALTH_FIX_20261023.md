# v3公开起点无路时的联合规划修正

## Linux故障与复现

用户在07a9625运行full_w64的pilot，NEW_B2第20回合累计5410步；第21回合结束于累计5436步，随后在map_contract.metrics触发continuous coupled-search health contract异常。该回合26步，并非单回合累计5436步。PyTorch创建张量的提示是性能警告，与这次异常不同。

现有failure.json只保存通用健康错误，没有失败条件计数；地图失败日志原先仅覆盖advance异常，未覆盖回合末metrics异常。

用相同训练生成器2002729重新生成第21项定义（没有加载旧模型或旧结果）。其搜索器0在初始**公开安全图**上的起点被标为invalid_start，公开几何point_free为false；搜索器1/2为ready。这不表示真实初始位置落在真实障碍物内，是已观测占据与安全净空约束下暂时没有安全连接。原D2也为搜索器0生成0条路径，将其安全保持在原位，为1/2规划。

旧v3却要求0/1/2全有候选，遇到0缺路径就将整个联合提案退回D2。缩短至32步、零残差的独立复现中，回合在25步因碰撞正常结束，滤波一直有效、24个应评分步均已评分，但8次规划回退、没有成功的新地图提案，最终健康检查因此拒绝该回合。这不是原Linux已学习策略的逐步重放；初始化不依赖模型，后续轨迹仅作为有限机制复现。

## 修正内容

- 只对v3，用同一公开过滤图证明起点invalid_start或no_start_connector，且该搜索器此前没有语义任务时，才把它作为不可行约束排除出当前候选分组。原SafetyBridge仍给出UNASSIGNED/hold，其他可达机器人继续做原联合打分、候选压缩与选择。没有构造虚假路径，也没有强迫机器人穿过障碍物。
- 若起点ready但候选生成失败，仍报错；已有任务不会被这个规则丢弃；所有分组均不可行时仍明确失败。局部恢复仍要求每个受影响搜索器得到有效任务，否则原子拒绝，保留原分配。
- v2保留原严格缺路径行为；原D2、物理和学习器文件保持字节一致。v3滤波、长期地图、窗口、粒子与几何内核均保持字节一致。4个载体、M=3、N=32、H=16、W=64及原候选/查询预算未缩小。
- **所有健康检查条件均保留**，包括至少一次成功提案、逐步有效性和评分数量。失败异常新增具体条件名，并保存map_failure_*.json，含阶段、步数、公共会话计数、滤波摘要、推断耗时和最近20条规划审计。不会把模型权重或隐藏真实障碍物写入该日志。

修正后的同一短场景零残差复现走完32步：8次新地图提案、0次D2回退、31个应评分步全部完成，过滤器有效。它支持这个路径约束问题已被修正，不代表原Linux第21回合或30回合训练已被完整复测。

## 更新并保留原RUN

原20回合、episode20检查点和失败记录保留原身份。原检查点resume_supported=false，未保存完整Replay恢复状态；不能把诊断权重当作可继续训练的原生检查点，也不能给它重标新源码。此修正需要新的训练RUN，本轮不自动执行长期训练或恢复模型文件。

在已有Linux克隆与AUV环境中执行。REV替换为聊天中给出的本次修复完整提交SHA；不能继续填07a9625。

```bash
conda activate AUV
REV='替换为本次修复的完整40位SHA'
git fetch origin && git checkout --detach "$REV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CHECK_FIX="$PWD/work/verification/route_health_linux_$STAMP"
python -m tools.validate_checkout --output "$CHECK_FIX" --expected-commit "$REV" --require-linux --execute
```

207项测试无失败/跳过，且validation.json最外层passed、linux_execution、map_workflow_verified为true后：

```bash
RUN_FIX="$PWD/work/collision_terminal/bounded_pilot_route_fix_$STAMP"
python -m cli bounded-experiment prepare --root "$RUN_FIX" --stage pilot --execute
python -m cli bounded-experiment run --root "$RUN_FIX" --arms full_w64 --validation "$CHECK_FIX/validation.json" --execute
```

新RUN仍是每个学习方法1种子×30回合，不修改训练超参数。其他六项条件的分阶段启动与汇总沿用GITHUB_LINUX.md，将root改为RUN_FIX即可。

若又出现异常，保留RUN_FIX，读取jobs/NEW_B2_seed2729/train/failure.json及map_failure_*.json，不覆盖原输出、不直接重复整个run。新的日志会具体显示no_core_proposal、missing_core_scored_steps或证据有效性条件。

## 来源与工程验证

发布的07a9625源码、配置与证据已冻结为map_bounded_20261023_v3父版。3.08MiB新基线仅保存非ZIP文件；8份既有归档使用不可变哈希引用，没有再次嵌套复制大ZIP。新来源通过map_route_health_manifest.json、map_route_health_evolution.json及verification/map_route_health_v1/<源码身份>/记录。原清单和证据不改写。独立验证器检查完整修改覆盖和受保护文件字节，当前检查库存207项。

本轮207项回归全部通过，无失败/跳过；另有B2/B3两个临时模型各完成一次32步训练、4次完整更新，分别直接用内存模型评估32步，权重保持不变。加上修正前后短复现及两次仅初始化诊断，本轮共8次回合初始化、6个非空短回合、185个物理步，未恢复检查点文件。第一次源演化负控检查遇到Windows长路径限制，已修正临时目录创建与清理的路径处理并完整重跑；初次失败与最终通过日志均保留。诊断脚本有一次初始化后的读取错误，也计入这8次初始化，没有为它额外重复回合。

最终证据位于verification/map_route_health_v1/8a589ad6dfebb47e75aca38714365ff7e9d2be11be23927bfa0dd4ba0261d255/。短复现是在最终清单生成前运行，三个功能修改文件此后未再变更；该证据单独记录功能文件绑定，不将它冒充为原Linux已学习模型的重放。新来源下的B2/B3临时训练评估另有独立凭据。本轮未重新运行旧640批计时，因为本次滤波和几何内核字节未变；旧计时保留旧身份。
