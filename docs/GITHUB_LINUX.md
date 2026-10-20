# GitHub 上传与 Linux 实验

当前新旧地图各自适应后的 D2+RL 完整操作已集中到 [新旧地图 Linux 实验指南](MAP_EXPERIMENT_LINUX_20261020_v1.md)。

该指南包含 Windows 检查/上传、固定提交克隆、依赖安装、当前 Linux 验收、开发集排除、六组独立训练、原生评估及统计汇总。请使用其中的 scripts/linux_map_experiment.sh 和 cli map-suite；原 cli suite 继续仅表示原地图三方法。

当前验收库存为 127 项，以 configs/rebuild_check_contract.json 为准。Linux 验收使用 --map-workflow 的八回合临时新模型检查，不恢复模型文件；Windows 随包证据不能代替 Linux 实机结果。

原地图三方法的旧训练入口仍见 [TRAINING](TRAINING.md)。本文件此前的 38 项测试、旧 --rollout 流程和原三方法 pilot 说明属于历史版本，已冻结在 provenance/baselines/map_core_20261020_v1/ 中，不作为当前地图六组实验指令。
