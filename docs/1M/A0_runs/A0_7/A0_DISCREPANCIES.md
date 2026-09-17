# A0.7 差异与处理

| 事项 | 事实与处理 |
|---|---|
| 全指标完整性 | 982 项中 2 项缺原始采集事件证据；其余各项有重算/核验或明确无定义、禁用、停止规则状态。全部 seed、质量及成本复算通过。 |
| 当前与旧最终质量 | 新 gold@10 均值 0.2967679791，旧 0.3031382168；变化按原精度列入结果表。 |
| 原始事件日志 | API 页数和重复丢弃次数无法由保留的去重快照恢复；文件名检索仅找到本轮阶段监控事件，不属于原始采集日志。 |
| 历史合并来源 | 本轮重新构造监督/节点/冲突三表，逐列逐行严格一致；按 A0 合同接受新来源计数。 |
| 计时环境 | 旧 Windows/Intel 与新 watgpu308/Linux 环境不同；分开展示，不将差值单独归因于七列修复。 |
| 精确参照耗时 | exact full 和 exact1000 是同一次共享扫描；总成本计一次。 |
| 监督进程终点 | 复算报告已完成，但监督进程的 OS 退出码/结束时间在中途切换任务后未记录；补充校验从现有原始文件与报告独立确认。 |
| 本机路径 | 原生产 metadata/protocol/manifest 保持字节；目录映射及旧英文不可变副本见 PATH_ALIASES.json 与 library_revision/REPLAY_PATH_MAP.json。 |

全文 evidence library 的后续更新以 A0.1–A0.7 为依据；旧结果仅作为 A0 已冻结的历史对照保留。
## A0.7 report unit correction

See library_revision/UNIT_CORRECTION.md and REPORT_REGENERATION.json: source fractions unchanged; declared percentages and old/new differences corrected. All 982 raw values/statuses preserved; 39 tests, 470 evaluator comparisons and 40 percentage-unit checks passed.
