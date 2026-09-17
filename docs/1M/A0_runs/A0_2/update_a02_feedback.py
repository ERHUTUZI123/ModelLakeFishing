"""Fill A0.2 completion feedback only after the final focused tests pass."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
DOCS = OUT.parent.parent


def main():
    tests = json.loads((OUT / 'final_focused_checks.json').read_text(encoding='utf-8'))
    assert tests['status'] == 'PASS'
    count = tests['counts']['tests']
    seconds = tests['seconds']
    finish = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-4))).strftime('%Y-%m-%d %H:%M:%S EDT')
    doc = DOCS / 'A0.2.md'
    text = doc.read_text(encoding='utf-8')
    assert '状态：**A0.2 已完成' not in text, 'One-time feedback writer; preserve completed delivery'
    text = text.replace('状态：**进行中，最终联合检查与证据归档待完成。** 开始时间：2026-09-12 20:46:16 EDT。',
        f'状态：**A0.2 已完成；A0.3–A0.7 待执行。** 开始时间：2026-09-12 20:46:16 EDT；完成回填：{finish}。')
    text = text.replace('已生成，独立复核中', '已生成；独立复核 98 项通过')
    text = text.replace('已实现，最终联合检查待完成', f'已实现；最终联合检查 {count} 项通过')
    text = text.replace('已实现，来源覆盖检查中', '已实现；618 项来源逐项登记，2 项原始事件证据缺口显式保留')
    text = text.replace('实际可执行命令及输出文件将在 [A0_COMMANDS.md](A0_runs/A0_COMMANDS.md) 的 A0.2 追加区登记。',
        '实际可执行命令及输出文件已在 [A0_COMMANDS.md](A0_runs/A0_COMMANDS.md) 的 A0.2 追加区登记。')
    text = text.replace('缺口详情与可执行来源覆盖将登记到', '缺口详情与可执行来源覆盖已登记到')
    text = text.replace('## 6. 执行反馈区',
        '来源覆盖统计：27 项读取 A0.1 本轮实际审计，15 项使用已实现的延后纯读 source recount，565 项由新运行原始记录及其汇总产生，3 项按原配置禁用，6 项按定义无全库排名而无定义，2 项缺原始事件证据。该统计是入口覆盖，**本步生成的新训练/检索指标数为 0**。\n\n'
        '## 6. 实际检查与执行反馈\n\n'
        f'最终联合测试 **{count} passed，0 failed，0 error，0 skipped**；包装器总耗时 {seconds:.3f} 秒。源码在测试前后哈希一致。覆盖七列/元数据、smoke/恢复、checkpoint/export、评测/复算、sidecar、同分/浮点边界、来源计数和原有 Y2/Y4 相关行为。没有运行 A0.3 的全目录回归或真实全规模 smoke。\n\n'
        '7 个 CLI 的 `--help` 均成功；40 个有效训练配置键与 A0.1 完全一致。测试有依赖弃用和只读 mmap 提示，详见完整日志；这些提示未导致失败。\n\n'
        '| 证据 | 内容 |\n|---|---|\n'
        '| [最终联合检查](A0_runs/A0_2/final_focused_checks.json)、[完整日志](A0_runs/A0_2/final_focused_tests.log)、[JUnit](A0_runs/A0_2/final_focused_tests.xml) | 精确命令、时间、退出码、通过数、测试前后源码哈希 |\n'
        '| [接口与配置检查](A0_runs/A0_2/A0_INTERFACE_CHECKS.json) | 7 个 CLI、40 键比对、实际本机环境 |\n'
        '| [独立新图/导出复核](A0_runs/A0_2/A0_GRAPH_REVIEW.json) | 98 项检查；初次失败和后续修正均保留 |\n'
        '| [历史源身份](A0_runs/A0_2/A0_HISTORICAL_SOURCE_IDENTITY.json) | 99 项检查；四份历史 SHA 证明缺口及后续输出一致性门 |\n'
        '| [A0.2 来源清单](A0_runs/A0_2/A0_2_MANIFEST.json)、[代码快照目录](A0_runs/A0_2/frozen/)、[代码明细](A0_runs/A0_2/A0_CODE_CAPTURE.json) | 最终源码归档、新图和文档哈希；保留 A0.1 入口快照 |\n'
        '| [交付核验](A0_runs/A0_2/A0_DELIVERY_VALIDATION.json) | 校验源码、归档和文档绑定，确认 A0 §1–§9 协议正文未改 |\n\n')
    text = text.replace('| 当前阶段 | A0.2 收尾中 |', f'| 当前阶段 | A0.2 已完成，{finish} |')
    text = text.replace('| 最终联合测试 | 待填最终日志及通过数 |', f'| 最终联合测试 | {count} 项通过，0 失败/错误/跳过；见上表 |')
    text = text.replace('| 独立新图复核 | 进行中 |', '| 独立新图复核 | 98 项通过；源图和最终图内容核验通过 |')
    text = text.replace('| 下一步 | 完成 A0.2 归档后，进入 A0.3 的全套回归、信息边界验证与真实 smoke |',
        '| 下一步 | A0.3：按 A0.md 原顺序执行全套回归、信息边界验证和真实 smoke；本步未启动 |')
    text = text.replace('最终完成时间、检查日志、完整代码快照、来源清单和交付核验结果将在本区回填。',
        'A0.2 的实现与新图验收完成。两个原始事件证据缺口继续留在反馈中；后续 A0.7 完整性检查在证据未补齐时必须非零退出，整个 A0 不能提前标记完成。')
    assert '进行中' not in text and '待填最终' not in text
    doc.write_text(text, encoding='utf-8')
    plan = DOCS / 'A0.md'
    text = plan.read_text(encoding='utf-8')
    text = text.replace('A0.1 已完成；A0.2 进行中；A0.3–A0.7 待执行，尚未提交训练或生成新检索结果。',
        'A0.1、A0.2 已完成；A0.3–A0.7 待执行，尚未提交训练或生成新检索结果。',1)
    text = text.replace('A0.1 已完成，证据见 [A0.1.md](A0.1.md)；**A0.2 进行中**，实时回填见 [A0.2.md](A0.2.md)。新图已生成，必要入口正在最终联合核验；正式训练及评估命令仍在其前置条件满足后执行。',
        'A0.1、A0.2 已完成，证据见 [A0.1.md](A0.1.md) 和 [A0.2.md](A0.2.md)。新图和必要入口已核验；下一步为 **A0.3**。下文“待实现”描述保留原计划语境，当前状态以反馈区为准。正式训练与评估仍须满足各阶段前置条件；两个上游事件计数的原始证据缺口见 A0.2 回填。',1)
    start = text.index('### 10.1 最新反馈')
    end = text.index('### 10.2 阶段进度', start)
    text = text[:start] + f'''### 10.1 最新反馈

| 项目 | 当前内容 |
|---|---|
| 最后更新 | {finish}，A0.2 完成 |
| 当前阶段 | A0.1、A0.2 已完成；A0.3–A0.7 待执行 |
| 本次已完成 | 七列修复与实际新图、必要评测/复算入口；{count} 项联合测试、98 项独立新图/导出复核通过，详见 [A0.2.md](A0.2.md) |
| 当前可做的下一步 | A0.3：原全套回归、信息边界测试和真实完整规模 smoke |
| 进入训练前还缺什么 | A0.3 测试和资源检查；三个正式 run 尚未启动 |
| 当前是否已有新结果 | **没有新训练/检索指标。新图已生成，所有正式指标仍待运行后计算。** |
| 正确性问题 | 七列/元数据/图绑定、smoke 隔离、A0 评测和精确参照一致性已实现；两个原始快照事件计数证据缺口仍保留 |
| 最终结论 | 待三个 seed 完整重算；原始事件缺口未补齐前，A0.7 全指标完整验收不能通过 |

''' + text[end:]
    text = text.replace('| A0.2 新图与实现适配 | 进行中 | 2026-09-12 20:46:16 EDT / — | [A0.2.md](A0.2.md) | 新图逐列/文件核验通过，最终联合检查中 | 完成测试与证据归档 |',
        f'| A0.2 新图与实现适配 | 已完成 | 2026-09-12 20:46:16 EDT / {finish} | [A0.2.md](A0.2.md)、[A0_2](A0_runs/A0_2/) | 新图七列/哈希验收通过；{count} 项联合测试、98 项独立复核通过 | A0.3 待执行 |')
    text = text.replace('| 快照与输入核验通过；旧参照/全指标登记齐全 | A0.2 待执行 |', '| 快照与输入核验通过；旧参照/全指标登记齐全 | A0.2 已完成 |')
    text = text.replace('| A0.2 实现，A0.3 验证 | 待处理 |','| A0.2 已实现并生成验证图；A0.3 补信息边界测试 | 已完成本步，见 A0.2 |')
    text = text.replace('| A0 模式保留定义、输出真实结果和门状态 | 待处理 |','| A0 模式保留定义、输出真实结果和门状态 | A0.2 已实现并通过 fixture |')
    text = text.replace('| 实现独立 smoke 选项，正式训练保持原入口 | 待处理 |','| 实现独立 smoke 选项，正式训练保持原入口 | A0.2 已实现；真实 smoke 待 A0.3 |')
    text = text.replace('| 同分 fixture 核验，按事实文档解决并记录 | 待核验 |','| 同分及 float32 probe/GEMM 一致性已修正并记录 | A0.2 fixture 通过；精确参照差异须注明修正 |')
    marker = '### 10.6 每次更新的简短记录'
    text = text.replace('\n\n' + marker,
        '\n| I-05 | A0.2 全指标来源复查 | 两个快照事件计数缺原始日志 | A0.7 全指标完整验收 | 保持 missing；不能复制旧 PROVENANCE 或删除指标 | 未解决，见 A0.2 §5 |\n'
        '| I-06 | A0.2 历史源身份复核 | 五源已找到；四份尚缺历史完整 SHA | 延后 source recount | 已绑定当前文件，完整重建 edge/node/conflict 三表一致后接受新计数 | 来源入口已实现，完整数据重算待执行 |\n\n' + marker,1)
    text += f'''\n\n时间：{finish}\n阶段：A0.2 完成。\n本次操作：生成最终七列修复图，保持 40 个有效训练配置键；完成必要训练/导出/评测/复算适配及证据归档。\n输出：A0.2.md、A0_runs/A0_2/；新图 digest acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db。\n已通过：最终联合测试 {count} 项，独立新图/导出复核 98 项，历史源身份复核 99 项，7 个 CLI。\n缺失：API 分页和重复丢弃事件日志；两项指标保持 missing，不影响已核验新图，但阻断后续全指标完整验收。\n下一步：按 A0.3 执行全套测试、信息边界检查及真实 smoke；本步未启动。\n'''
    plan.write_text(text,encoding='utf-8')
    print(json.dumps({'A0.2':'completed_feedback','test_count':count,'time':finish},ensure_ascii=False))


if __name__ == '__main__':
    main()
