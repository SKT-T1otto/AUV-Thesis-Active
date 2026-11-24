# Linux 开发先导实验

本批检验同一家族、同一续程下，改变两艘艇的出航指派是否改变原生终局成功事件。按 [冻结计划](plan.json) 依次处理 `0003 → 0020 → 0014`。每个家族取重复号 0、1、2、3，每个重复号采集 `00 / 01 / 10 / 11`，第三艘搜索艇取选项 0。三个家族均属于开发集。共有 12 个配对组、48 条终局，其中 `0003_r0` 的四条复用历史回执。本次最多新增 44 条、17,600 个物理步。

## 手动启动

在 Linux 机器进入本仓库根目录，激活项目的 Python 环境。环境依赖见 [requirements.txt](../../requirements.txt)。从 Windows 复制 `linux_inputs.zip` 到服务器任意目录。包内是冻结的策略权重，以及 `0003_r0` 的请求、四条分支与独立审计。若服务器已有这七个原文件，可以省略 `--inputs`。

```bash
git pull --ff-only origin main
python -B -m tools.run_launch_development_pilot --inputs ~/linux_inputs.zip --workers 2 --execute
```

去掉 `--execute` 只做预检。入口逐文件核对源码、清单、权重和历史结果的 SHA-256，并从历史逐步记录重建四条标签。正式启动后先在各家族时刻零重新生成候选；在 `0003` 还会和历史的物理初态、观测及候选逐项比对。若平台差异影响了冻结出航条件，入口会在相应检查处报错。可用 `--workers 1` 至 `4` 调整同组并行分支数。每个工作进程使用一个 Torch CPU 线程。此入口只采集和审计。

命令中断后，使用同一条命令继续。完成的分支按请求识别、按原始记录重审。尚未落盘的分支从该次出航重新开始。若目录已有不同请求、冲突的权重或不相符的审计记录，程序会指出冲突并停止。每个家族在续程之间使用相同的出航候选和实际初始指令；每个重复号内，四种指派共享按艇寻址的续程随机流。

若投影后有艇缺少计划要求的选项，程序会保存该家族的 `*_unavailable.json`，并继续其他可采家族。此时不能形成完整的三家族结论。已经能够提交、随后被原生接口拒绝或回退的指派照常进入原指派分母。

## 查看结果

新采集的原始分支与 `audit.json` 位于本目录的 `<家族编号>_r<重复号>/` 下。历史四条仍从 [原目录](../launch_interventions_v1/development_terminal) 读取。全部通过后，本目录生成 `summary.json`。可单独复核：

```bash
python -B -m tools.launch_development_pilot_audit
```

- `families[].by_arm` 给出各指派的观察数、发现数、收包数和原生成功数。每种指派每个家族预定四次，全零家族也列出。
- `success_discordant_blocks` 列出在同一家族、同一重复号中，部分指派成功、部分失败的组。`expand_paired_validation` 为真表示值得扩大配对验证。四次续程只能作为先导证据。
- 若至少两个家族各有超过半数分支始终未发现，`inspect_search_coverage` 为真，先检查出航路线与续程覆盖，再考虑拟合分工规则。该标记可以与扩大验证同时出现。
- 每条分支记录投影、实际接纳和路线；几何诊断给出首次续程前与整个原生探测期间扫到的无障碍网格中心数、最近探测距离减去探测半径的余量。这些事后诊断仅供解释发现阶段。主指标取原生终局成功，折扣回报从逐步团队奖励按 0.95 重算。

如需在持有原件的机器上重新导出输入包，可在仓库根目录运行：

```bash
python -B -m tools.run_launch_development_pilot --pack-inputs verification/launch_development_pilot_v1/linux_inputs.zip
```
