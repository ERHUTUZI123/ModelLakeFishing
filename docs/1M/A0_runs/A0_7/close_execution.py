from pathlib import Path
from datetime import datetime,timezone
import json,psutil
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
s=read(BASE/'STATUS.json');r=read(BASE/'results/A0_REPORT.json');v=read(BASE/'results/A07_VALIDATION.json')
assert not psutil.pid_exists(s['pid']) and v['retrieval_and_cost_recomputation']=='PASS'
s.update(status='incomplete' if not r['completeness']['complete'] else 'complete',phase='recomputed_and_validated',
         updated_at_utc=datetime.now(timezone.utc).isoformat(),completeness=r['completeness'],
         all_ann_fidelity_passed=r['all_ann_fidelity_passed'],additional_comparisons=v['additional_evaluator_comparisons'],
         supplemental_validation=v['status'])
s['steps']['recompute'].update(exit_code=None,
    completion_evidence='CLI log reports incomplete and all report/inventory outputs exist; supplemental checks PASS. Original supervisor ended during turn interruption, so OS exit code and supervision endpoint were not captured.',
    expected_cli_exit_from_completed_report=2 if not r['completeness']['complete'] else 0)
s['steps']['source_counts']['elapsed_scope']='Supervisor start-to-join interval, overlapping artifact downloads; not isolated source-recount CPU duration.'
(BASE/'STATUS.json').write_text(json.dumps(s,indent=2)+'\n')
commands=['# A0.7 实际命令与环境','',
    '执行主机：本机 i7-14650HX；Python 3.13.1；CPU 8 线程。训练和正式检索的原始耗时/环境取自 A0.4–A0.6 记录。','',
    '9 个完整导出/索引文件通过 scp 下载，原始 SHA 全部核验；实际列表见 DOWNLOAD_PLAN.json。','',
    '以下为实际 argv（JSON 数组，保留精确参数和 Windows 路径）：','']
for name,step in s['steps'].items():
    commands += ['## '+name,'','```json',json.dumps(step,indent=2,ensure_ascii=False),'```','']
commands += ['源统计与文件传输并行；source_counts 的监督区间包含等待传输完成的时间，不能当作独占运行耗时。',
    '用户中途转向更新 evidence library 时原监督进程终止，独立复算已写出完整报告；随后以 finalize_delivery.py 完成 470 项额外对照。操作系统退出码未采得，按未观测保留。',
    '复算程序的预期退出码 2 表示仍有缺失项；本轮确认为两项采集事件日志。',
    'finalize_delivery.py、backfill.py 在本阶段完成后执行；library_revision/preserve_evidence.py 为后续正文更新保存原文与可复现目录。','']
(BASE/'A0_COMMANDS.md').write_text('\n'.join(commands),encoding='utf-8')
(BASE/'A0_DISCREPANCIES.md').write_text('''# A0.7 差异与处理

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
''',encoding='utf-8')
print(s['status'],r['completeness']['inventory_items'],len(r['completeness']['missing']))
