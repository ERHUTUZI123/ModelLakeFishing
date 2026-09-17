"""Compare non-quality summaries too, then archive reviewable A0.7 tables."""
from pathlib import Path
import csv,hashlib,json,math,platform,sys
BASE=Path(__file__).resolve().parent
OUT=BASE/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    report=read(OUT/'A0_REPORT.json');inv=read(OUT/'A0_METRIC_INVENTORY.json')
    ev=read(BASE.parent/'A0_6/delivery/metrics/A0_EVALUATION_REPORT.json')
    values=report['new_measurements'];items={x['id']:x for x in inv['metrics']}
    checked=[]
    def equal(label,a,b):
        if a is None or b is None:ok=a is b
        else:ok=math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
        assert ok,(label,a,b)
        checked.append(label)
    def get(scope,name,seed):return values[f'{scope}.{name}.{seed}']['value']
    latency={'hnsw_latency_p50_ms':'hnsw_p50','hnsw_latency_p95_ms':'hnsw_p95',
        'prior_rerank_latency_p50_ms':'rerank_p50','prior_rerank_latency_p95_ms':'rerank_p95',
        'total_latency_p50_ms':'end_to_end_p50','total_latency_p95_ms':'end_to_end_p95'}
    for seed in range(3):
        row=ev['hnsw'][str(seed)]
        for name,key in latency.items():equal(f'latency.{name}.{seed}',get('retrieval_cost',name,seed),row['latency_ms'][key])
        for name in ['build_seconds','index_bytes','index_gib']:
            equal(f'index.{name}.{seed}',get('index_cost',name,seed),row[name])
        equal(f'ef.{seed}',get('ann_calibration','selected_ef_search',seed),row['ef_search'])
        equal(f'recall.{seed}',get('ann_calibration','selected_recall_at_1000',seed),row['recall@1000'])
        trace=get('ann_calibration','calibration_trace',seed)
        assert len(trace)==len(row['ef_trace'])
        for a,b in zip(trace,row['ef_trace']):
            equal(f'calibration.ef.{seed}.{a["ef_search"]}',a['ef_search'],b['ef_search'])
            equal(f'calibration.recall.{seed}.{a["ef_search"]}',a['recall'],b['recall@1000'])
        for scope,summary in [('hnsw1000_task_prior',ev['hnsw_summary']),
                              ('exact1000_task_prior',ev['summary']['G_exact1000_task']),
                              ('exact_full_lake_task_prior',ev['summary']['G_full_task'])]:
            for key,val in report['native_recomputed'][str(seed)][scope].items():
                equal(f'native_summary.{scope}.{key}.{seed}',val,summary[key]['per_seed'][seed])
        record=read(BASE.parent/f'A0_6/delivery/metrics/A0_EVAL_RECORDS_s{seed}.json')
        for name in ['exact_full_reference_seconds','exact1000_reference_seconds','complete_evaluation_seconds',
                     'evaluation_peak_rss_bytes','evaluation_peak_vram_bytes']:
            equal(f'pipeline.{name}.{seed}',get('pipeline_cost',name,seed),record[name])
    for name,key in latency.items():
        for agg in ['mean','min','max']:
            equal(f'latency_summary.{name}.{agg}',get('retrieval_cost',name,agg),ev['latency_summary_ms'][key][agg])
    for name in ['exact_pool_retention','ann_retention','overall_retention']:
        saved=ev['decision'][name]
        for seed in range(3):
            item=values[f'retrieval_diagnostics.{name}.{seed}']
            equal(f'{name}.{seed}',item['value'],saved['per_seed'][seed])
            equal(f'{name}.numerator.{seed}',item['numerator'],saved['numerators'][seed])
            equal(f'{name}.denominator.{seed}',item['denominator'],saved['denominators'][seed])
        for agg in ['mean','min','max']:equal(f'{name}.{agg}',get('retrieval_diagnostics',name,agg),saved[agg])
    for name in ['build_seconds','index_bytes','index_gib']:
        for agg in ['mean','min','max','sum']:
            equal(f'index_summary.{name}.{agg}',get('index_cost',name,agg),ev['index_cost'][name][agg])
    for scope,saved in [('hnsw1000_task_prior',ev['hnsw_summary']),
                         ('exact1000_task_prior',ev['summary']['G_exact1000_task']),
                         ('exact_full_lake_task_prior',ev['summary']['G_full_task'])]:
        for name,entry in saved.items():
            vals=[report['native_recomputed'][str(s)][scope][name] for s in range(3)]
            for agg,fn in [('mean',lambda x:sum(x)/len(x)),('min',min),('max',max)]:
                equal(f'quality_summary.{scope}.{name}.{agg}',None if any(v is None for v in vals) else fn(vals),entry[agg])
    allowed={'frozen_input_audit.model_snapshot_api_pages.shared','frozen_input_audit.model_snapshot_skipped_duplicates.shared'}
    missing=set(report['completeness']['missing'])
    assert not report['completeness']['invalid'] and not report['completeness']['missing_seeds']
    assert missing<=allowed,sorted(missing-allowed)
    assert all(get('reproducibility_checks','independent_metric_recomputation',s) is True for s in range(3))
    unit_checks=[]
    for item in inv['metrics']:
        rec=values.get(item['id'],{})
        raw=rec.get('value');reported=item.get('new_value')
        if item.get('unit')=='percent' and type(raw) in (int,float):
            assert rec.get('unit') in ('fraction','percent'),(item['id'],rec.get('unit'))
            expected=raw*100 if rec['unit']=='fraction' else raw
            assert math.isclose(reported,expected,rel_tol=1e-12,abs_tol=1e-12),(item['id'],raw,reported)
            unit_checks.append(item['id'])
    compared={x['id']:x for x in report['comparisons']}
    for item in inv['metrics']:
        old,new=item.get('old_value'),item.get('new_value')
        if type(old) in (int,float) and type(new) in (int,float):
            assert math.isclose(compared[item['id']]['difference_from_published_value'],new-old,rel_tol=1e-12,abs_tol=1e-12),item['id']
    proof={'status':'PASS_WITH_MISSING_SOURCE_EVENTS' if missing else 'PASS',
           'retrieval_and_cost_recomputation':'PASS','additional_evaluator_comparisons':len(checked),
           'comparison_keys':checked,'unit_contract_checks':unit_checks,'complete':report['completeness']['complete'],
           'missing':sorted(missing),'invalid':report['completeness']['invalid'],
           'report_sha256':sha(OUT/'A0_REPORT.json'),'inventory_sha256':sha(OUT/'A0_METRIC_INVENTORY.json'),
           'local_environment':{'python':sys.version,'platform':platform.platform()},
           'measured_environment':ev['environment']}
    write(OUT/'A07_VALIDATION.json',proof)
    # Full inventory is retained; scalar old/new/delta entries also get a compact table file.
    columns=['id','scope','seed','name','unit','status','old_value','new_value','delta_new_minus_old','old_display_precision','old_source_section','new_reason']
    with (OUT/'A0_COMPARISONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for x in inv['metrics']:
            a,b=x.get('old_value'),x.get('new_value')
            rec={k:x.get(k) for k in columns}
            rec['delta_new_minus_old']=b-a if type(a) in (int,float) and type(b) in (int,float) else None
            for k,v in list(rec.items()):
                if isinstance(v,(dict,list)):rec[k]=json.dumps(v,ensure_ascii=False)
            writer.writerow(rec)
    text=(OUT/'A0_RESULTS.md').read_text(encoding='utf-8')
    marker='\n## A0.7 补充验收与完整对照\n'
    if marker in text:text=text.split(marker)[0]
    def display(v,empty='不适用'):
        if v is None:return empty
        return f'{v:.10g}' if isinstance(v,float) else str(v)
    def cell(item,old=False):
        if old:return display(item.get('old_value'),'未报告')
        status=item.get('status','not_applicable')
        return display(item.get('new_value'),{'undefined':'无定义','disabled':'禁用','missing':'缺失','not_applicable':'不适用'}.get(status,'缺失'))
    lines=[marker,'',f'独立质量计分通过；另外 {len(checked)} 项计时、成本、校准及汇总对照通过。',
        '逐项清单及所有原始未舍入值以 JSON 为准；[A0_COMPARISONS.csv](A0_COMPARISONS.csv) 保存全清单旧值、新值、差值和原显示精度。','',
        '### 最终系统逐 seed 新旧对照','',
        '| 指标 | seed | 旧值 | 新值 | 差值（新−旧，原单位） |','|---|---|---:|---:|---:|']
    for name in ['gold_at_1','gold_at_10','top3_at_10','gold_gap_at_10','root_macro_gold_at_10',
                 'median_gold_position_when_retrieved','actual_hnsw_gold_coverage_at_1000']:
        for seed in [0,1,2,'mean']:
            x=items.get(f'hnsw1000_task_prior.{name}.{seed}')
            if not x:continue
            a,b=x.get('old_value'),x.get('new_value')
            delta=b-a if type(a) in (int,float) and type(b) in (int,float) else None
            label=name+('（%）' if x.get('unit')=='percent' else '')
            lines.append(f'| {label} | {seed} | {cell(x,True)} | {cell(x)} | {display(delta)} |')
    for title,scope,names in [('ANN 校准','ann_calibration',['selected_ef_search','selected_recall_at_1000','calibration_passed']),
        ('索引与完整流水线成本','index_cost',['build_seconds','index_bytes','index_gib']),
        ('本轮训练、导出与评测成本','pipeline_cost',['training_seconds','export_seconds','prior_build_seconds',
         'exact_full_reference_seconds','exact1000_reference_seconds','complete_evaluation_seconds',
         'training_peak_rss_bytes','training_peak_vram_bytes','evaluation_peak_rss_bytes','evaluation_peak_vram_bytes'])]:
        lines += ['',f'### {title}','','| 指标 | seed 0 | seed 1 | seed 2 | 新均值 | 旧均值 |','|---|---:|---:|---:|---:|---:|']
        for name in names:
            line=[name]+[cell(items.get(f'{scope}.{name}.{s}',{})) for s in [0,1,2,'mean']]
            line += [cell(items.get(f'{scope}.{name}.mean',{}),True)]
            lines.append('| '+' | '.join(line)+' |')
    anchor=items['hnsw1000_task_prior.gold_at_10.mean']
    changes=[(items[f'hnsw1000_task_prior.gold_at_10.{s}']['new_value']-items[f'hnsw1000_task_prior.gold_at_10.{s}']['old_value'])*100 for s in range(3)]
    conclusion=f"修复后最终 gold@10 三 seed 平均为 {anchor['new_value']:.10f}，旧值为 {anchor['old_value']:.10f}，变化 {(anchor['new_value']-anchor['old_value'])*100:+.4f} 个百分点。三个 seed 分别变化 "+'、'.join(f'{x:+.4f}' for x in changes)+' 个百分点；保留率和 ANN 精度使用本轮精确参照重新计算。'
    total_bytes=get('index_cost','index_bytes','sum');total_gib=get('index_cost','index_gib','sum')
    lines += ['', f'三份索引合计 **{total_bytes:,.0f} bytes = {total_gib:.10f} GiB**。',
        '两项 exact 参照耗时来自同一次共享扫描，成本求和时计一次。资源峰值沿用记录的进程生命周期范围；索引大小为真实磁盘字节数。',
        '训练 25 个 epoch 的所有原生字段和损失序列均保存在 A0_REPORT.json 与逐项清单中，禁用的诊断保留 disabled。',
        '', '### 结论与适用范围','',
        conclusion,
        '旧延迟来自 Windows 11 / Intel / 24 逻辑处理器；本轮正式延迟来自 watgpu308 / Linux / 32 逻辑处理器。表中并列新旧实测值，硬件差异使其不能单独用于判断特征修复的性能影响。本机 A0.7 复核耗时另行记录。',
        '评估对象为冻结节点表中的 dataset–task 节点，性能边在两个方向留出。4,122 是三个 split 的查询观察次数；结果衡量已有历史最优模型的排名恢复。新节点编码及未观测模型–查询组合的下游表现仍需另行实验。',
        '七个性能派生输入列已归零；修复范围和其余输入保持情况有独立字节校验。初始化固定为 0，三 seed 反映数据划分变化。',
        '', '### 完整性反馈','',
        '全部检索、成本及可恢复来源统计已复算。API 分页次数与重复丢弃次数仍缺原始采集事件日志，因此完整性验收保留缺口。补齐可验证的原始日志后，应为两项计数增加来源证据并重新执行完整性验收。','']
    (OUT/'A0_RESULTS.md').write_text(text+'\n'.join(lines),encoding='utf-8')
    state=read(BASE/'STATUS.json');state['supplemental_validation']=proof['status'];state['additional_comparisons']=len(checked)
    write(BASE/'STATUS.json',state)
    print(json.dumps({k:proof[k] for k in ['status','additional_evaluator_comparisons','missing']},ensure_ascii=False))
if __name__=='__main__':main()
