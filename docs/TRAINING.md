# 在 Active 内手动训练和评估

本目录对应母版当前D2 repair v2。D2执行原先验与导航，残差恒为0；D2+B2和D2+B3训练原MADDPG及B3边界编码器。
HGR不在当前方案中。当前默认配置为1000回合、每回合最多400步，可能因碰撞或成功提前结束。
它与母版当前repair-v2配置一致；不自动套用旧Linux四臂Suite的400000环境步/固定训练池协议。

## 一、环境和来源

```powershell
conda activate AUV
Set-Location E:\gym\code\WORKSPACE\AUV-Thesis-Active
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m cli verify
```

来源必须匹配，`manual_research_execution_allowed`必须为true才能开始非机制训练。
这表示当前重建的工程验收允许用户手动运行，不表示论文结果或历史E0已经通过。
每次代码/配置来源变化都需要对应的新来源与检查记录，不能把旧checkpoint改名伪装成新版本。

## 二、推荐的D2、B2、B3三方法流程

以下命令全部由用户手动执行。场景导出会生成数据；`suite prepare`只建立固定计划，不开始训练。
三种方法共用同一份有序独立评估场景，三个训练seed产生六个训练任务，D2只评估一次。

```powershell
$STAMP = Get-Date -Format 'yyyyMMdd_HHmmss'
$BASE = Join-Path $PWD 'work\collision_terminal'
$EVAL = Join-Path $BASE "evaluation_$STAMP.json"
$RUN = Join-Path $BASE "study_$STAMP"

python -m cli scenarios --split eval --count 100 --seed 92729 --output "$EVAL" --execute

python -m cli suite prepare --root "$RUN" --manifest "$EVAL" --seeds 2729 3729 4729 --execute

python -m cli suite run --root "$RUN" --stage train --arms D2_B2 D2_B3 --execute

python -m cli suite run --root "$RUN" --stage evaluate --execute

python -m cli suite summarize --root "$RUN" --output "$RUN\report" --execute
```

只做D2与D2+B2时，prepare增加 `--arms D2 D2_B2`，train只选择 `--arms D2_B2`。
`suite run --stage all --execute`可按计划先训练再评估，但不会由工具或验证脚本自动替用户执行。

计划固定本次源码、评估清单、seed和配置文件的哈希。默认训练沿用母版repair-v2按回合生成场景的采样器，
并非自动改成固定训练池。评估会检查与实际训练场景的ID、seed和物理fingerprint是否重叠。
若要研究其他预算或固定训练池，须显式建立相应配置与实验记录。

## 三、单独运行一个训练任务

```powershell
python -m cli train --config configs/b2.json --seed 2729 --output work/collision_terminal/b2_seed2729 --execute
python -m cli train --config configs/b3.json --seed 2729 --output work/collision_terminal/b3_seed2729 --execute
```

两个目录都必须是新的或为空。训练用全新模型；最终检查点为 `checkpoint_final.pt`，每100回合另存一次。
训练日志和checkpoint记录实际环境步数、Actor/Critic优化器步数、配置和来源身份。
恢复历史Replay/断点续训不属于母版当前repair-v2支持的接口，本次未另行引入。

## 四、单独评估D2和一个当前检查点

```powershell
python -m cli reference --config configs/b2.json --manifest "$EVAL" --output work/collision_terminal/d2_evaluation --execute

python -m cli evaluate --checkpoint work/collision_terminal/b2_seed2729/checkpoint_final.pt --manifest "$EVAL" --output work/collision_terminal/b2_evaluation --execute

python -m cli summarize --episodes work/collision_terminal/d2_evaluation/episodes.jsonl work/collision_terminal/b2_evaluation/episodes.jsonl --output work/collision_terminal/comparison --execute
```

单独汇总的完整性仅针对显式提供的评估文件；suite汇总则对照计划判断缺失的seed/方法。
输出 `report.md`、`groups.csv`、`summary.json` 和 `ordering.json`。
Found、Pre-Found collision、失败惩罚Found时间与Success保持原指标含义；失败时间惩罚为本次horizon。
条件成功时间不可替代发现率，D2重复用于配对比较不增加独立样本数。

每个评估完成时重新核验来源并签发带文件哈希的 `evaluation_identity.json`。
没有完成凭据、源码变化、清单变化或结果文件被改动时，结果汇总会拒绝。

## 五、运行状态与中断

用 `python -m cli suite run --root "$RUN" --stage evaluate` 可查看计划，不加载模型或开始评估。
同一个suite禁止并发写入。正常退出会清理自己的 `.active_suite.lock`；强制终止留下的锁需要先确认所属进程状态，
不要直接删锁来绕过仍在运行的任务。已有完整训练凭据的任务可跳过；未完成且已有内容的输出不会被覆盖或自动恢复。
失败后应保留现场并使用新的输出目录重跑。

## 六、有界工程检查

```powershell
python -m tools.verify_source
python -m tools.run_checks --output work/verification/check_01 --execute --rollout
```

rollout模式只用新模型和有界fixture：每模型最多4轮多智能体更新、每场景最多8步。
为了触发更新，fixture临时使用batch/warmup=4/4；正式配置仍为128/256。
机制检查点默认只能用于加 `--mechanism-check` 的有界评估，不能作为正常研究训练结果混入汇总。
