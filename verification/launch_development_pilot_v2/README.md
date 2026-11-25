# 在同一运行环境采集 48 条终局

2026-11-25，Linux 在时刻零重建 0003 候选时，与 Windows 历史记录出现最大约 1.1920929e-6 米的坐标差异。用户报告两端 NumPy 均为 2.2.6，Torch 均为 2.11.0+cu126。旧入口的精确候选比较因而停止，尚未进入 Linux 新终局分支。具体数值差异可能来自平台或 CPU 数学库，尚未定位到单个算子。

用户已确认在 Linux 重建重复号 0。此修订从原计划派生，复用相同的三个开发家族、候选生成器、策略权重和随机流。48 条分支全部在同一运行环境采集，最多 19,200 个物理步。原 Windows 四条结果在原目录保留为历史参照。

## 预检和运行

在 Linux 仓库根目录、Conda AUV 环境中运行。

```bash
python -u -B -m tools.run_launch_development_pilot --runtime-local
```

此命令验证源码与历史输入，生成三个家族的候选，并把物理初态、候选、抽样记录、随机流状态和运行环境写入本目录的 runtime.json。物理推进步数为零。已有 runtime.json 时逐项精确核对。历史与本机的候选差异也会记录在其中。

第一次运行应显示 48 条计划终局、0 条历史终局计入本批、48 条回执待采，随后出现三个家族的候选检查信息。预检完成后运行。

```bash
nohup python -u -B -m tools.run_launch_development_pilot +  --runtime-local --workers 2 --execute +  >> launch_pilot_v2.log 2>&1 < /dev/null &
echo "PID=$!"
```

当前目录的 launch_pilot_v2.log 记录进度。使用同一条命令续跑。原有 linux_inputs.zip 可以继续使用，若原文件尚未导入，可在预检时添加 --inputs 指向该包。

## 配对与验收

0003、0020、0014 依次执行。每家族包含重复号 0、1、2、3，各重复号包含 00、01、10、11。第三艘搜索艇固定选项 0。每组使用相同的续程流，各重复号间的续程流独立。

本批所有分支的候选、初态、抽样记录和原始/额外候选随机流必须精确匹配本机 runtime.json。同一指派跨重复号的出航指令、任务根和路线号也须一致。独立审计复算终局标签，发现、收包和成功分别统计。

原生拒绝和回退保留在提交指派分母中。候选投影后缺少所需选项的家族保存 unavailable 回执，并继续其他可采家族；完整批次结论需全部计划组通过。

全部完成后本目录生成 summary.json。可独立复核。

```bash
python -B -m tools.launch_development_pilot_audit +  --folder verification/launch_development_pilot_v2
```

结果的 archived_terminals_pooled 为 0，historical_reference 指向原 Windows 回执及其哈希。每种指派的成功数、同重复号的成功不一致组和搜索覆盖诊断沿用原先导计划的计算方式。
