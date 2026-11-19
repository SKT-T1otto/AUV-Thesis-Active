# 0000紧急刹车审查与旧成功场回归

控制实现保持任务同步提交ba494e2，原task-sync源码身份aadffb932cfd53e2b134623a7fb38b01ffe4edcbfa7ed0d027b721750fd0853f不变。本次没有确认足以改变控制律或安全阈值的接口缺陷，没有修改旧源码/清单。新增工具和只读诊断由独立regression manifest固定；旧runtime身份不能替代新诊断代码的检查，新诊断不能把旧结果换成新实验标签。

## 0000已确认和未知

trace step144正常刹车、第145步应急刹车，其归一化动作与原brake纯函数误差均小于1e-7；next-speed诊断也匹配。前一物理步的实际速度/加速度可以复算公开flow估计，下一步实际速度与固定当前flow模型的误差小于.001m/s，支持动作/单位/执行衔接正常。

第145步所有普通动作和刹车的有限预测都因motion_or_backup_geometry失败，原代码显式开放应急刹车，非新同步代码跳过安全检查。第146步导航恢复，第147—148步采用原已验证退出动作，第149步恢复普通导航。审查窗口138—153步的实际运动线段离真实AABB最小约.6043米，未碰撞；真值只用于离线审计，未用于控制。没有碰撞不等于获得形式安全保证。

available_distance由第144步约.535降至第145步约.094，但缺少当时hit、子体素及粗地图快照，无法唯一判定观测收紧、粗代理或缓存问题。因此分类是“继承应急逻辑按记录触发，未发现动作接口缺陷，几何拒绝根因REVIEW_REQUIRED”。不因此降低padding、扩大速度cap或换地图。

## 只读拒绝诊断

新增diagnostics.py代理只调用原geometry.segment并原样返回布尔值，原guard.check的结果/metadata原样返回。对拒绝段另记预测起止点、extra/padding、公开几何分支、阻塞子体素、未知段数和邻近扫描hit；不查询真值、不调用环境step/路径规划/RNG、不改变观测动作契约。只为非冻结的navigation_blocked或emergency部署步保留详细记录，最近两步内存缓存有界。

还记录该查询起点的hit/coarse距离、unknown/free、物理/几何时钟、map revision、last_scan_step与扫描/子体素cursor；粗几何拒绝时保留实际公开占据盒及该预测段逐子体素free/blocked标记。它们是查询当时的局部证据，不是从真值虚构全图，可用于区分新观测收紧、子体素边界和过期视图。

诊断异常仅记REVIEW_REQUIRED，不能改变原控制结果。分类使用既有公开几何：bounds、blocked_subcell、unobserved_coarse_geometry、scan_hit_padding。记录命中分支，不虚构完整逐步地图。诊断含额外几何查询/序列化成本，回归wall_seconds不作为受控速度基准。这个观察器没有恢复当年0000缺失的地图，也不会自动重跑0000。

## 本次Linux回归范围

仅0003、0004、0005，新版本各1回合，每场最多400物理步，总计最多1200步；规则控制、不训练、不加载权重。使用原运行seed982004/982005/982006及episode index3/4/5，不能从0重新编号改变配对。A组导入原tracking both三场成功轨迹，不重跑A。

运行时控制源码未变，因此复用用户上次真实Linux验证（59项、4×32步）作为同一控制版本的依据；读取原收据并逐附件核验，不修改其source或plan标签。新增诊断有独立来源及合成检查，在Linux运行新增检查时物理步0；本命令不追加旧128步准备，也不重复0000—0002。

```bash
conda activate AUV
cd ~/AUV-Thesis/AUV-Thesis-R123
(
  set -euo pipefail
  git pull --ff-only
  OLD=work/collision_terminal/r123_tracking_20261117_121047_2475513
  VALIDATED=work/collision_terminal/r123_task_sync_20261118_223936_3078331
  BASE="work/collision_terminal/r123_task_sync_regression_$(date +%Y%m%d_%H%M%S)_$$"
  test -n "$OLD" && test -n "$VALIDATED" && test -n "$BASE"
  bash scripts/r123_task_sync_regression_linux.sh "$OLD" "$VALIDATED" "$BASE"
)
```

路径非空检查、输入哈希检查、全新输出和set -euo pipefail确保遇错停止。session.env记录BASE，新终端先从实际目录恢复；不要读取不存在的latest文件或用sudo绕过路径问题。每25步进度，checks/prepare/evaluate/report日志都保留。只有sync/completed.json及report/report.json均存在并打印ALL THREE REGRESSION SCENES COMPLETED才算本工具完成。

新原始episode含failure_diagnostics.records。汇总比较成功/发现/碰撞、发现/成功时刻、接触/保持、低速/刹车/等待和根任务完成，并单列紧急回退和拒绝分支。旧baseline的task_accepts从原始事件归一到搜索器口径，不使用四智能体旧总数直接比较。

回传整个BASE目录后再作决定：若丢失旧成功或出现碰撞，先定位回归，不合并成4/6成功；若没有明显退步，保留同步修复，将后续排查转到0000/0002搜索相遇和剩余低速，每次只改一个有证据的问题，不直接启动网络训练或批量矩阵。
