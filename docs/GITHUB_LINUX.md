# GitHub 版本与 Linux 运行

当前地图方案为 `coupled_search_bounded_v3`，默认原始窗口64步。实现、七组配置和指标解释见 [有界地图实现与实验](MAP_BOUNDED_V3_20261022.md)。此前 map-suite 六组入口仍保留为 v2 实验；本版有界地图使用 `cli bounded-experiment`。

## 安装与当前版本验证

使用新目录克隆。聊天中给出的完整提交 SHA 是本次发布版本；将下面占位符替换为该 SHA。不要在有旧输出的目录内重置工作树。

```bash
set -euo pipefail
EXPECTED_COMMIT='替换为本次发布的完整40位SHA'
git clone --config core.autocrlf=false https://github.com/SKT-T1otto/AUV-Thesis-Active.git AUV-Thesis-Active-bounded-v3
cd AUV-Thesis-Active-bounded-v3
git checkout --detach "$EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"

conda create -n AUV-bounded-v3 python=3.10 -y
conda activate AUV-bounded-v3
python -m pip install -r requirements.txt
python -m pip check
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CHECK="$PWD/work/verification/bounded_linux_$STAMP"
python -m tools.verify_source
python -m cli verify
python -m tools.validate_checkout --output "$CHECK" --expected-commit "$EXPECTED_COMMIT" --require-linux --execute
VALIDATION="$CHECK/validation.json"
cat "$VALIDATION"
```

验证结果应有 `passed=true`、`linux_execution=true`、`map_workflow_verified=true`，192项测试无失败/跳过。实际库存以 configs/rebuild_check_contract.json 为准。验证最多执行8回合×32步、4个临时模型各4次完整更新，不恢复检查点文件；W16只用于短回合窗口滑动检查。Windows证据不能代替这次Linux实测。

## 先做 pilot

以下命令会实际训练。pilot每个地图条件独立训练B2/B3，1个种子、30回合、每回合最多400步，6个固定验证任务、12个开发评估任务；ZERO不训练。先运行完整方案及v2参考。失败条件保留输出，继续运行其他条件，以免只报告成功项。

```bash
RUN="$PWD/work/collision_terminal/bounded_pilot_$STAMP"
python -m cli bounded-experiment prepare --root "$RUN" --stage pilot --execute
for arm in full_w64 reference_v2; do
  if ! python -m cli bounded-experiment run --root "$RUN" --arms "$arm" --validation "$VALIDATION" --execute; then
    echo "Incomplete arm retained: $arm"
  fi
done
```

需要比较窗口和机制时，再运行五项对照：

```bash
for arm in full_w32 full_w128 no_boundary_w64 no_local_recovery_w64 window_map_w64; do
  if ! python -m cli bounded-experiment run --root "$RUN" --arms "$arm" --validation "$VALIDATION" --execute; then
    echo "Incomplete arm retained: $arm"
  fi
done
REPORT="$PWD/work/bounded_pilot_report_$STAMP"
python -m cli bounded-experiment report --root "$RUN" --output "$REPORT" --execute
cat "$REPORT/report.md"
```

只运行两项时，报告会把剩余五项标为未完成，不宣称七项实验完整。`arms.csv`记录完成状态，`resources.csv`记录推断耗时与有界状态/恢复工作，`paired_deltas.csv`只比较已完成的原生策略；逐项任务结果和学习曲线保存在各条件的报告子目录。

## 正式阶段

pilot分析后，使用全新目录准备 `--stage formal`：2个训练种子、300回合、12个固定验证任务、100个最终评估任务。然后按上面的run/report入口运行选定条件。开发集和最终集生成种子分离；最终集不用来选窗口或参数。

```bash
FORMAL="$PWD/work/collision_terminal/bounded_formal_$STAMP"
python -m cli bounded-experiment prepare --root "$FORMAL" --stage formal --execute
# 后续 run 使用 --root "$FORMAL"，report 使用全新的输出目录。
```

所有输出位于work/，不会改变发行源码或Git干净状态。每次重新验收/准备/汇总使用新路径；已有RUN只能按同一计划继续未完成阶段。更换提交后须重新Linux验收并准备新RUN。源码验证与包清单要求文件字节完全一致，请保留 .gitattributes，不转换LF/CRLF。不带 `--execute` 的执行入口只显示参数。本版仍为研究实现，不代表Linux性能已通过验收或强化学习必然改善结果。
