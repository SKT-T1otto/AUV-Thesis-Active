# 搜索失败审计：GitHub交付与Linux离线复核

本次交付包含tracking结果核查、搜索失败检查、分配—执行接口检查三份审计及其复算脚本；新增跨平台入口tools.r123_search_audit和Shell封装。不包含承诺同步、候选池或路点控制的行为修复。生产运行代码仍继承42f2add，tracking源码身份仍为1b51edf677adca45266d8e5ecf818f060f08d8a7ec0b2d476710ce8d127f6326。新增审计入口具有自己的脚本哈希，不给原运行代码或旧实验换标签。

前三个verification目录中的Windows复算脚本依赖当时的目录布局和快照；Linux请使用下述新入口。原始3090结果没有上传GitHub；Linux复用已有work目录，不必复制Windows的3090结果。

## Linux命令

在AUV环境已可用的终端运行。此命令检查四组共24条**既有**episode，不创建环境、不跑新回合、不训练、不加载模型，新增物理步上限为0。不会重跑已完成tracking三组。

```bash
conda activate AUV
cd ~/AUV-Thesis/AUV-Thesis-R123
(
  set -euo pipefail
  git pull --ff-only
  RUN=work/collision_terminal/r123_tracking_20261117_121047_2475513
  OLD=work/collision_terminal/r123_governed_20261116_215327_3078331/governed
  OUT="work/collision_terminal/r123_search_audit_$(date +%Y%m%d_%H%M%S)_$$"
  test -n "$RUN" && test -n "$OLD" && test -n "$OUT"
  test -f "$RUN/plan/plan.json"
  test -f "$RUN/both/completed.json"
  test -f "$OLD/completed.json"
  bash scripts/r123_search_audit_linux.sh "$RUN" "$OLD" "$OUT"
)
```

子shell启用遇错停止和pipefail；git pull失败不会继续；BASE不会从不存在的latest文件读取；输出必须是全新目录。原始输入、完成记录与当前源码身份均核验后才计算，逐组逐场打印进度。日志在输出目录同级的`输出目录.log`，完成标志为输出目录内completed.json，另有REPORT.md、analysis.json和输入保留核验。

新终端查看时，先把上次打印的真实输出路径赋给OUT，例如：

```bash
OUT='work/collision_terminal/r123_search_audit_实际时间_实际编号'
test -n "$OUT" && test -f "$OUT/completed.json" && cat "$OUT/REPORT.md"
```

不要直接执行含“实际时间”占位文字的示例；使用脚本最终打印的路径。没有completed.json表示未通过，检查同级log或输出中的failed.json；不要用sudo绕过路径/输入问题。

## 预期结果与边界

成功数baseline/path_only/feedforward_only/both依次2/2/3/3，发现均3，碰撞均0；低速搜索器步数440/319/440/319。核对目标扫掠发现时刻、原始结果哈希、既有源码身份、低速cap复算与输入未变。

这些是旧结果的离线复核，不是修复效果评估、新训练或新Linux源码准入。该入口不新增当时缺失的公开地图、候选池及全部路线快照；也没有实施承诺回传。后续若要验证接口修复，需先完成独立实现与针对性验证，再提供新的有限评估命令，不能现在把both旧组重复运行称作修复对照。
