# AUV-Thesis-Active：D2 / D2+B2 / D2+B3

当前入口与训练流程见 **[训练和评估指南](docs/TRAINING.md)**。
本目录重建的是原 AUV-Thesis 当前工作树的 D2 repair v2，实现行为保持一致；
训练、场景导出、D2参考评估、B2/B3检查点评估和结果汇总均在本目录完成。

```powershell
conda activate AUV
Set-Location E:\gym\code\WORKSPACE\AUV-Thesis-Active
python -m cli verify
```

`verify` 显示来源及本版本的手动研究运行准入。所有执行入口缺省只打印计划，用户加 `--execute` 才运行。
默认B2/B3配置、网络、原复合奖励、gamma=.95、4智能体及28D/3D/124D保持母版行为。
D2是零RL残差对照，不训练神经网络。来源身份独立，不直接接收旧目录checkpoint。

当前重建证据见 [REBUILD](docs/REBUILD.md)，逐文件职责见 [CODE_MAP](docs/CODE_MAP.md)。
已知奖励/障碍问题见 [KNOWN_LIMITS](docs/KNOWN_LIMITS.md)；重建不保证强化学习提升性能。
历史E0验收、正式性能结论均未通过本次重建转移。

