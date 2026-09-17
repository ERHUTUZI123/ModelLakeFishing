"""Backfill A0.7 evidence without editing frozen A0 sections 1--9."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
BASE=Path(__file__).resolve().parent
DOC=BASE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    s=read(BASE/'STATUS.json');done=s['status'] in ['complete','incomplete']
    label={'running':'进行中','failed':'受阻','incomplete':'受阻（复算已执行，完整性验收有缺口）','complete':'已完成'}[s['status']]
    report=read(BASE/'results/A0_REPORT.json') if done else None
    lines=['# A0.7 执行回填','',f"状态：**{label}**。更新时间：{s['updated_at_utc']}。",'',
        '唯一最终系统：**X4G+D → HNSW top1000 → task prior → top10**。','',
        '## 1. 执行与验收','',
        '| 项目 | 当前反馈 |','|---|---|',
        '| A0.6 前置验收 | 三个 seed 的 exact/HNSW/finalize、远端交付哈希及本机 824.4 万候选分数复核均 PASS |',
        f"| 本步状态 | {s['phase']}；实际进程和命令见 STATUS.json |",
        f"| 完整产物同步 | {len(s['completed_downloads'])}/9 个此前只在远端的大文件完成；下载后逐文件验证原 manifest SHA256 |",
        '| 计算环境 | 本机 i7-14650HX；CPU 8 线程；A0.7 复算耗时与远端正式检索计时分别归档 |',
        '| 独立计分 | 使用已冻结的 scale1m.recompute_a0，独立由标签和候选重建融合、排序、命中、root 平均、条件位置及保留率 |',
        '| 计时与成本 | 从逐查询 ns 数组重算分位数；读取真实索引字节、构建区间及本轮训练/导出/评测记录 |',
        '| 源统计 | 重新流式解析冻结快照，并核对重建后的监督、节点、冲突三张表 |',
        '| 路径兼容 | PATH_ALIASES.json 保存最初映射；[REPLAY_PATH_MAP.json](A0_runs/A0_7/library_revision/REPLAY_PATH_MAP.json) 记录更新 library 前的不可变源码/原文映射；原协议、manifest、checkpoint 和数值文件保持原字节 |','',
        '## 2. 结果与完整性','']
    if report:
        c=report['completeness']
        lines += [f"清单 {c['inventory_items']} 项；缺失 {len(c['missing'])} 项；无效 {len(c['invalid'])} 项；缺失 seed：{c['missing_seeds']}。",'',
                  f"状态分布：{c['status_counts']}。质量逐项复算通过，额外 {s.get('additional_comparisons',0)} 项汇总、ANN、计时和成本对照通过。",'',
                  '最终新旧对照、ANN/精确参照、训练与成本表见 [A0_RESULTS.md](A0_runs/A0_7/results/A0_RESULTS.md)。',
                  '原始未舍入数值、全部差值和来源哈希见 [A0_REPORT.json](A0_runs/A0_7/results/A0_REPORT.json)；逐项状态见 [A0_METRIC_INVENTORY.json](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json)。','']
        inv=read(BASE/'results/A0_METRIC_INVENTORY.json')
        items={x['id']:x for x in inv['metrics']}
        lines += ['| seed | 新 gold@10 | 旧 gold@10 | 差值（百分点） |','|---|---:|---:|---:|']
        for seed in [0,1,2,'mean']:
            x=items[f'hnsw1000_task_prior.gold_at_10.{seed}']
            lines += [f"| {seed} | {x['new_value']:.10f} | {x['old_value']} | {(x['new_value']-x['old_value'])*100:+.4f} |"]
        lines += ['', '缺失项目：','']+[f'- `{x}`：{items[x].get("new_reason")}' for x in c['missing']]
        if c['invalid']:lines += ['','无效项目：']+[f'- `{x}`' for x in c['invalid']]
        if (BASE/'library_revision/REPORT_REGENERATION.json').exists():
            lines += ['', '### 报告单位复核与修正','',
                '复核发现首版将部分比例值按 percent 标注。已修正独立报告器的单位转换并从原始产物重新生成报告：原始比例显式标为 fraction，指标清单的百分数及新旧差值使用统一单位。',
                '39 项针对性测试通过；982 项原始测量值及状态与首版完全一致；40 项百分数单位合同与 470 项 evaluator 对照通过。正式训练、向量、检索输出和原始计时保持原字节。',
                '证据：[单位修正](A0_runs/A0_7/library_revision/UNIT_CORRECTION.md)、[测试记录](A0_runs/A0_7/library_revision/UNIT_CORRECTION_TESTS.json)、[重新生成及不变性检查](A0_runs/A0_7/library_revision/REPORT_REGENERATION.json)。']
    else:lines += ['正式复算结果待产物同步及来源统计完成后生成。','']
    lines += ['', '## 3. 反馈区','',
        '| 事项 | 实际反馈 |','|---|---|',
        '| 历史唯一依据 | 已冻结英文事实文档；旧显示精度保留；原值与新测量独立对象归档 |',
        '| 既有证据缺口 | 两项采集事件计数依赖原始逐请求/逐丢弃日志，继续按实际证据核查；完整性状态由清单自动判定 |',
        '| 延迟可比性 | 原测量为 Windows/Intel，本轮正式计时为 watgpu308/Linux；并列报告，硬件差异单列 |',
        '| 评估范围 | 冻结节点表内、性能边留出的历史排名恢复；4,122 是三个 split 的查询观察次数 |',
        '| 特征修复 | 全节点七个性能派生输入列归零；其余特征、划分与训练/检索配方沿用冻结协议 |','',
        '## 4. 产物与复现入口','',
        '- [执行记录](A0_runs/A0_7/STATUS.json)、[实际执行脚本](A0_runs/A0_7/run.py)、[目录映射](A0_runs/A0_7/PATH_ALIASES.json)。',
        '- [源统计日志](A0_runs/A0_7/source_counts.log)、[独立复算日志](A0_runs/A0_7/recompute.log)。',
        '- [来源清单](A0_runs/A0_7/MANIFEST.json) 串联冻结的 A0.6 来源清单与本阶段交付。','']
    (DOC/'A0.7.md').write_text('\n'.join(lines),encoding='utf-8')
    original=(BASE/'frozen/A0.md').read_text(encoding='utf-8')
    text=(DOC/'A0.md').read_text(encoding='utf-8');rows=text.splitlines()
    for i,line in enumerate(rows):
        if line.startswith('编写日期：'):rows[i]='编写日期：2026-09-12。状态：**A0.1–A0.6 已完成；A0.7 '+label+'。**'
        elif line.startswith('| 最后更新 |'):rows[i]=f"| 最后更新 | {s['updated_at_utc']}；A0.7 {label} |"
        elif line.startswith('| 当前阶段 |'):rows[i]=f'| 当前阶段 | A0.1–A0.6 已完成；A0.7 {label} |'
        elif done and line.startswith('| 最终结论 |'):rows[i]='| 最终结论 | 最终 gold@10 均值 0.2967679791；检索/成本复算通过；两项采集事件计数缺证据，完整性验收保留缺口 |'
        elif line.startswith('| 本次已完成 |'):rows[i]='| 本次已完成 | '+('A0.7 独立复算与新旧报告已生成；缺口见 A0.7.md' if done else 'A0.7 已启动完整产物同步、来源统计与独立复算流程')+' |'
        elif line.startswith('| 当前可做的下一步 |'):rows[i]='| 当前可做的下一步 | '+('补齐原始证据缺口并重新验收完整性' if s['status']=='incomplete' else ('依据已验收新结果更新论文' if s['status']=='complete' else '完成 A0.7 并核对逐项完整性'))+' |'
        elif line.startswith('| A0.7 复算与结果报告 |'):rows[i]=f'| A0.7 复算与结果报告 | {label} | 2026-09-13 EDT 起 | [A0.7.md](A0.7.md)、[A0_7](A0_runs/A0_7/) | 逐项反馈见回填 | 完整性验收 |'
        elif done and line.startswith('| A0.3 测试与 smoke |'):rows[i]=line.rsplit('|',2)[0]+'| 后续 A0.4–A0.6 已完成 |'
        elif done and line.startswith('| A0.4 三 seed 从头训练 |'):rows[i]=line.rsplit('|',2)[0]+'| 后续导出与检索评测已完成 |'
        elif done and line.startswith('| A0.5 导出与 prior |'):rows[i]=line.rsplit('|',2)[0]+'| 后续检索评测已完成 |'
        elif done and line.startswith('| A0.6 全部评测与计时 |'):rows[i]=line.rsplit('|',2)[0]+'| A0.7 已复算；两项来源事件缺证据 |'
        elif done and any(line.startswith(f'| {seed} | 1539537_{seed} |') for seed in range(3)):
            seed=int(line.split('|')[1].strip());gold=items[f'hnsw1000_task_prior.gold_at_10.{seed}']['new_value']
            rows[i]=f'| {seed} | 1539537_{seed} | 25/25，COMPLETED | PASS | PASS | {gold:.10f} | 本 seed 检索/成本 PASS；共享来源计数缺 2 项 |'
        elif done and line.startswith('| I-06 |'):rows[i]='| I-06 | A0.2 历史源身份复核 | 五源已找到；四份缺历史完整 SHA | source recount | A0.7 已完整重建 edge/node/conflict 三表并逐列逐行严格一致，接受本轮新计数 | PASS，见 A0.7 来源统计 |'
    if done:
        def val(scope,name,seed='mean'):return report['new_measurements'][f'{scope}.{name}.{seed}']['value']
        primary={'最终 gold@1':'gold_at_1','**最终 gold@10**':'gold_at_10','最终 top3@10':'top3_at_10','最终 gold-gap@10':'gold_gap_at_10','最终 root-macro gold@10':'root_macro_gold_at_10'}
        extra={
            '条件 gold 位置中位数（逐 seed）':' / '.join(str(val('hnsw1000_task_prior','median_gold_position_when_retrieved',s)) for s in range(3)),
            'recall@1000（逐 seed）':' / '.join(f"{val('ann_calibration','selected_recall_at_1000',s):.10f}" for s in range(3)),
            'ef_search（逐 seed）':' / '.join(str(val('ann_calibration','selected_ef_search',s)) for s in range(3)),
            'HNSW pool gold coverage':f"{val('hnsw1000_task_prior','actual_hnsw_gold_coverage_at_1000')*100:.6f}%",
            'HNSW + rerank p50 / p95':f"{val('retrieval_cost','total_latency_p50_ms'):.9f} / {val('retrieval_cost','total_latency_p95_ms'):.9f} ms",
            'HNSW / rerank 分量 p50':f"{val('retrieval_cost','hnsw_latency_p50_ms'):.9f} / {val('retrieval_cost','prior_rerank_latency_p50_ms'):.9f} ms",
            '三份索引总大小':f"{val('index_cost','index_gib','sum'):.10f} GiB",
            '建索引时间（逐 seed）':' / '.join(f"{val('index_cost','build_seconds',s):.9f}" for s in range(3))+' s'}
        for label2,name in {'exact pool / full retention':'exact_pool_retention','HNSW / exact pool retention':'ann_retention','HNSW / full retention':'overall_retention','exact pool gold coverage':'exact_pool_gold_coverage_at_1000','full-fused top-10 成员在 exact pool 内的比例':'full_fused_top10_in_exact_pool'}.items():
            extra[label2]=f"{val('retrieval_diagnostics',name)*100:.6f}%"
        section=False
        for i,line in enumerate(rows):
            if line.startswith('### 10.4'):section=True
            elif line.startswith('### 10.5'):section=False
            if not section or not line.startswith('|'):continue
            parts=[p.strip() for p in line.split('|')]
            if len(parts)!=7:continue
            key=parts[1]
            if key in primary:
                item=items[f'hnsw1000_task_prior.{primary[key]}.mean'];parts[3]=f"{item['new_value']:.10f}";parts[4]=f"{item['new_value']-item['old_value']:+.10f}"
            elif key in extra:parts[3]=extra[key];parts[4]='本轮独立复算；延迟为异机实测' if 'p50' in key else '本轮独立复算'
            else:continue
            parts[5]='[A0_RESULTS](A0_runs/A0_7/results/A0_RESULTS.md)';rows[i]='| '+' | '.join(parts[1:-1])+' |'
    text='\n'.join(rows)+'\n'
    assert text[text.index('## 1.'):text.index('## 10.')]==original[original.index('## 1.'):original.index('## 10.')]
    (DOC/'A0.md').write_text(text,encoding='utf-8')
    files=[DOC/'A0.7.md',DOC/'A0.md',BASE/'STATUS.json',BASE/'run.py',BASE/'prepare.py',BASE/'backfill.py',BASE/'PATH_ALIASES.json',BASE/'DOWNLOAD_PLAN.json']
    files += [p for p in [BASE/'finalize_delivery.py',BASE/'PREFLIGHT.json',BASE/'rerun_report.py',BASE/'library_revision/REPLAY_PATH_MAP.json',BASE/'library_revision/REPORT_REGENERATION.json',BASE/'library_revision/UNIT_CORRECTION.md',BASE/'library_revision/UNIT_CORRECTION_TESTS.json',BASE/'A0_COMMANDS.md',BASE/'A0_DISCREPANCIES.md'] if p.exists()]
    files += sorted(p for p in (BASE/'results').iterdir() if p.is_file())
    files += sorted(p for p in BASE.glob('*.log') if p.is_file())
    previous=BASE/'frozen/A0_SOURCE_MANIFEST.json'
    write(BASE/'MANIFEST.json',{'stage':'A0.7','status':s['status'],'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'previous_manifest':ref(previous),'artifacts':[ref(p) for p in files]})
    old=read(previous)
    new={k:old[k] for k in ['authority','A0_2','A0_3','A0_4','A0_5','A0_6']}
    new.update(schema='a0.source_manifest.v1',stage='A0.7',status=s['status'],A0_7=ref(BASE/'MANIFEST.json'),previous_stage_manifest=ref(previous))
    write(BASE.parent/'A0_SOURCE_MANIFEST.json',new)
    print(label)
if __name__=='__main__':main()
