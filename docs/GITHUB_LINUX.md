# 新 GitHub 仓库与 Linux 验证

本指南只操作 `AUV-Thesis-Active`。母版 AUV-Thesis、历史输出和旧 checkpoint 留在原目录。
先做工程验证，再手动决定是否做性能试验；工程通过不代表 RL 必然优于 D2。

## Windows：首次上传

在 GitHub 建立一个新的空仓库，例如 `SKT-T1otto/AUV-Thesis-Active`。
不要勾选 README、license 或 .gitignore。仓库名可自行改变，下面 URL 必须随之修改。
这是 GitHub 的[现有本地代码上传流程](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。
私有库可通过 Git Credential Manager/浏览器授权；不要把 token 写入脚本、URL 或代码。

以下整段只用于尚未初始化 Git 的 Active 目录。函数确保某一步失败后停止。

```powershell
conda activate AUV
Set-Location E:\gym\code\WORKSPACE\AUV-Thesis-Active
$ErrorActionPreference = 'Stop'
function GitOK { & git @args; if ($LASTEXITCODE -ne 0) { throw 'Git 失败，停止。' } }
function PythonOK { & python @args; if ($LASTEXITCODE -ne 0) { throw 'Python 检查失败，停止。' } }
$Remote = 'https://github.com/SKT-T1otto/AUV-Thesis-Active.git'
if (Test-Path .git) { throw '本目录已有 Git；先核对 git status 和 git remote -v，不要重复初始化。' }

PythonOK -m tools.verify_source
PythonOK -m cli verify
GitOK init -b main
GitOK config core.autocrlf false
GitOK add -- .gitattributes .gitignore AGENTS.md PACKAGE_SHA256.json README.md cli.py project_paths.py requirements.txt core chapter3_bser configs checks tools provenance verification docs
PythonOK -m tools.validate_checkout --index-only --execute
GitOK diff --cached --stat
GitOK commit -m 'Import standalone Active D2 repair-v2 research project'
GitOK remote add origin $Remote
GitOK push -u origin main
$Commit = (GitOK rev-parse HEAD).Trim()
$RemoteHead = ((GitOK ls-remote origin refs/heads/main) -split '\s+')[0]
if ($RemoteHead -ne $Commit) { throw '远端提交不匹配。' }
Write-Host "上传验证成功；Linux 必须使用此完整 SHA：$Commit"
```

如果 Git 要求姓名/邮箱，先为这个新仓库设置 `git config user.name` 和 `git config user.email`，再从 commit 继续。
若仍需要之前 Windows 的本地代理，只重试 push 和远端校验，不要重新 commit：

```powershell
GitOK -c http.proxy=http://127.0.0.1:7897 -c http.version=HTTP/1.1 push -u origin main
GitOK -c http.proxy=http://127.0.0.1:7897 -c http.version=HTTP/1.1 ls-remote origin refs/heads/main
GitOK rev-parse HEAD
```

最后两行 SHA 必须一致。GitHub 密码不用于 HTTPS Git 认证；使用已有凭据/浏览器授权或 PAT。

## Linux：固定提交、安装环境、工程验证

替换 `EXPECTED_COMMIT` 为 Windows 最后打印的 40 位 SHA，仓库 URL 也要对应实际新库。
目录与环境名称均用于新建，不覆盖旧实验。私有仓库首次克隆需已授权的 Git 凭据或 SSH。
Windows 的 `127.0.0.1:7897` 代理不要原样带到 Linux。

```bash
set -euo pipefail
EXPECTED_COMMIT='在这里粘贴Windows打印的完整40位SHA'
git clone https://github.com/SKT-T1otto/AUV-Thesis-Active.git
cd AUV-Thesis-Active
git checkout --detach "$EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"

conda create -n AUV-Active python=3.10 -y
conda activate AUV-Active
python -m pip install -r requirements.txt
python -m pip check
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m tools.verify_source
python -m cli verify

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CHECK="work/verification/linux_$STAMP"
python -m tools.validate_checkout --output "$CHECK" --require-linux --expected-commit "$EXPECTED_COMMIT" --execute
cat "$CHECK/validation.json"
```

必须看到最外层 `passed: true`、`linux_execution: true`、正确 HEAD、38 项测试且无 failure/error/skip。
命令返回非零或证据缺失时停止，保留日志；不要删 manifest 或修改哈希来让它通过。
`validation.json`、`checks.stdout.log`、`checks.stderr.log` 和 `checks/` 下是这台 Linux 的新结果。
Windows 随包报告只代表之前本地验证，不会自动算作 Linux 通过。

验证使用全新临时模型，B2/B3 各最多 4 轮完整更新，实际新增物理环境步数为 35。
它检查来源/包字节、Git 干净检出、学习更新、Replay、冻结 Actor、B3 context、检查点、
真实环境短轨迹及 D2/B2/B3 汇总闭环。为触发更新，fixture 的 batch/warmup 为 4/4；正式配置仍为 128/256。
不能用这些机制结果比较算法优劣，也不运行论文长训练。

## Linux：手动小规模性能试验

**仅在上述 Linux 工程验证成功后执行。** 下面采用 20 回合/seed、3 个 seed、50 个独立评估场景。
只缩短回合预算与保存间隔，400 步 horizon、奖励、gamma、网络、学习率、warmup 和调度不变。
三方法共 6 个训练任务，理论训练上限 48000 环境步；完整评估上限 140000 步，可能运行较久。
这是探索性 pilot；20 回合可能不足以学习，不能据此认定 B2/B3 的算法原理不成立。

先建立输入和固定计划，以下步骤不会启动训练：

```bash
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE="$PWD/work/collision_terminal"
INPUT="$BASE/pilot_inputs_$STAMP"
RUN="$BASE/pilot_$STAMP"
mkdir -p "$BASE"
mkdir "$INPUT"

python - "$INPUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for label in ('b2', 'b3'):
    c = json.loads(Path(f'configs/{label}.json').read_text(encoding='utf-8'))
    c['total_main_trajectories'] = 20
    c['checkpoint_interval'] = 20
    c['max_total_environment_steps'] = None
    (out / f'{label}.json').write_text(json.dumps(c, indent=2), encoding='utf-8')
PY

python -m cli scenarios --split eval --count 50 --seed 92729 --max-steps 400 --output "$INPUT/evaluation.json" --execute
python -m cli suite prepare --root "$RUN" --manifest "$INPUT/evaluation.json" --b2-config "$INPUT/b2.json" --b3-config "$INPUT/b3.json" --seeds 2729 3729 4729 --execute
python -m cli suite run --root "$RUN" --stage train
printf '本次RUN=%s\n' "$RUN"
```

需要只比较 D2 与 D2+B2 时，prepare 加 `--arms D2 D2_B2`，后面的训练 arms 改成 `D2_B2`。
确认 plan.json 的方法、预算和路径后，再手动启动训练：

```bash
python -m cli suite run --root "$RUN" --stage train --arms D2_B2 D2_B3 --execute
```

训练全部完成后，手动评估并汇总：

```bash
python -m cli suite run --root "$RUN" --stage evaluate --execute
python -m cli suite summarize --root "$RUN" --output "$RUN/report" --execute
cat "$RUN/report/report.md"
cat "$RUN/report/groups.csv"

python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
for path in sorted((Path(sys.argv[1]) / 'jobs').glob('*/train/summary.json')):
    s = json.loads(path.read_text(encoding='utf-8'))
    print(path.parent.parent.name, 'episodes=', s['episodes'], 'steps=', s['environment_steps'],
          'updates=', s['optimizer_updates'], 'actor_steps=', s['actor_optimizer_steps'])
    if s['optimizer_updates'] == 0 or not any(s['actor_optimizer_steps']):
        raise SystemExit('至少一个模型未实际学习；本次结果不能代表完成训练后的算法性能。')
PY
```

D2 只评估一次。比较 Found、pre-Found collision、失败惩罚 Found 时间、Success 以及训练 seed 间差异，
不要仅比较回报或成功条件下时间。当前 checkpoint 使用最终完成回合，不按测试集挑最好模型。
同一套评估场景共享随机创新；训练/评估 ID、seed 和物理 fingerprint 都进行隔离检查。
如果数据量不足或训练尚未稳定，需要预先确定下一轮预算并使用新的输出目录，不能拼接旧曲线或静默恢复旧 checkpoint。

默认完整规模（1000 回合/seed）请使用 [TRAINING](TRAINING.md) 的原配置和新的 suite 计划，
不要把 pilot 目录改名当作正式实验。本次任务没有执行任何这样的训练。

## 中断、后台与证据

前台运行便于第一轮定位问题。若使用 tmux，应在新会话中重新激活环境并设置同一 RUN。
会话断开不会改变任务的来源、预算或身份。强行中止会留下未完成输出；不要直接删除锁文件或覆盖重跑。
先检查锁的 host/PID 和实际进程；失败现场保留，另建新 RUN。
训练未完成且没有完成凭据的目录不会被自动恢复。

`.gitattributes` 使用 `-text` 保留原始文件字节，这是 [Git 的正式属性语义](https://git-scm.com/docs/gitattributes)。
来源/历史证据包含合法的 CRLF 字节，不能在上传或 Linux 拉取时自动改成 LF。
新增 Python 文件可采用 LF，但修改后仍须建立对应来源版本；不要批量转换历史文件。
