
## A0.2 追加反馈

A0.1 以上各行保持当时结论；本段记录后续处理，完整证据见 `../A0.2.md` 和 `A0_2/`。

| 编号 | 本步发现 / 处理 | 对结果的影响与状态 |
|---|---|---|
| D09 | 已实际生成七列清零图；其他图文件保持原字节，原元数据关键编码字段保持一致并纳入新 digest | D01 的输入修复已完成；A0.3 信息边界与真实 smoke 仍待执行 |
| D10 | 首版 prepared graph 的 digest 未包含关键编码元数据；保存首版后修正、重建并固定最终 digest。剥离 marker、修复文件或显式 formal_training_eligible=false 均不能绕过对应检查 | 初始发现与复核修正均保留；只使用最终图。源输入未改 |
| D11 | full-fused top-10 的任意同分选择与计分排名矛盾；A0 使用原 label-free tie 顺序保存 top-10 | 精确参照一致性修正；legacy 保留原行为 |
| D12 | 128 维同向量 fixture 复现 probe 逐元素归约与全库 GEMM 的 float32 舍入不同；A0 probe 使用同形状分块 GEMM | 不改精度/融合公式，无容差或舍入排序。精确参照增加离线计算；新旧 exact 差异不能全部归因于特征修复 |
| D13 | 增加正式 run/export/prior/eval 的生产端 SHA 和 native 时间/资源记录；完成状态、epoch、history、run 身份须匹配，分块一致性验证须实际执行且通过 | 记录与绑定适配，不改变模型训练或最终 K=1000 系统计算。真实数据尚未运行 |
| D14 | D04 的 A0 模式和 smoke-only、D05 的相关小型绑定/tie 检查已实现；效果下降和 ef 全失败保存原判定后完成测量 | 不将失败门强制改为 true，不改阈值，不扩 K 或训练臂 |
| D15 | 首轮查找漏掉 repo/stage1BuildTransferGraph，扩大查找后找到五个历史源；99 项身份复核通过。ModelLens 有旧完整 SHA，其余四源目前为新冻结 SHA 和结构一致证据 | 撤回“历史五源文件缺失”的初步判断；后续纯读 merge 须重建全部 canonical edge/node/conflict 输出并核对一致，再接受新计数 |
| D16 | model_snapshot_api_pages 和 model_snapshot_skipped_duplicates 缺原始逐请求/逐丢弃事件证据；去重 shards 和旧 PROVENANCE 无法逆推 | 两项保持 missing，阻止 A0.7 全指标完整验收；不复制旧计数，不删指标或改合同 |
| D17 | 上游事件计数缺口由 A0.2 逐项来源复查发现；A0.1 已保存的事实/协议/库存不回写成另一套历史状态 | 新证据与缺口在本步追加。新图验收与后续全指标验收分别记录 |
| D18 | PowerShell 管道将文档中文追加转成问号，中文状态替换未匹配；终检发现后改用 UTF-8 文件直接写入，逐段复核 | 错误版本留存；仅文档修正。代码、图和 189 项通过结果未受影响 |
