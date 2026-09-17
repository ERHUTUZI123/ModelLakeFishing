"""Independent source-table and inventory-contract review for A0.1."""
import hashlib
import json
import re
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
OUT = AUDIT.parent
SOURCE = OUT.parent / 'EVIDENCE_SOURCE_LIBRARY_en.md'
doc = SOURCE.read_text(encoding='utf-8')
old = json.loads((OUT/'A0_OLD_REFERENCE.json').read_text(encoding='utf-8'))
inventory = json.loads((OUT/'A0_METRIC_INVENTORY.json').read_text(encoding='utf-8'))
protocol = json.loads((OUT/'A0_PROTOCOL.json').read_text(encoding='utf-8'))
checks = []

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check(name, condition, details=None):
    checks.append({'name':name,'passed':bool(condition),'details':details})

sections = {}
headers = list(re.finditer(r'^#{2,3} (\d+(?:\.\d+)?)\.?(?: |$)',doc,re.M))
for i,h in enumerate(headers):
    end = headers[i+1].start() if i+1<len(headers) else len(doc)
    sections[h.group(1)] = doc[h.start():end]

def table(section, header_start):
    lines = sections[section].splitlines()
    start = next(i for i,line in enumerate(lines) if line.startswith(header_start))
    out=[]
    for line in lines[start+2:]:
        if not line.startswith('|'): break
        out.append([x.strip().replace('**','').replace('`','') for x in line.strip('|').split('|')])
    return out

def number(x): return float(x.replace(',','').replace(' ms','').strip())
ref_map={(x['scope'],x['name'],x['seed']):x for x in old['records']}
independent={}
quality=table('5.4','| Split seed | Queries |')
names=['queries','gold_at_1','gold_at_10','top3_at_10','gold_gap_at_10','root_macro_gold_at_10','median_gold_position_when_retrieved']
for row in quality:
    seed='mean' if row[0]=='Three-seed mean' else int(row[0])
    for name,value in zip(names,row[1:]):
        if value!='—': independent[('hnsw1000_task_prior',name,seed)]=number(value)
raw=re.search(r'The unrounded final values are ([0-9.]+), ([0-9.]+), and ([0-9.]+); their arithmetic mean is \*\*([0-9.]+)\*\*',sections['5.4'])
check('source_contains_all_four_high_precision_final_values',raw is not None)
for s,v in zip([0,1,2,'mean'],raw.groups()):
    independent[('hnsw1000_task_prior','gold_at_10',s)]=float(v)
    r=ref_map[('hnsw1000_task_prior','gold_at_10',s)]
    table_row=quality[s if isinstance(s,int) else 3]
    check(f'final_gold10_both_displays_seed_{s}',r['published_literal']==v and r['other_published_display']['literal']==table_row[3]
          and r['display_precision']=='10 decimal places')
paths=table('5.4','| Retrieval path |')
for row,scope in zip(paths,['exact_full_lake_task_prior','exact1000_task_prior','hnsw1000_task_prior']):
    for seed,value in zip([0,1,2,'mean'],row[2:]):
        if scope!='hnsw1000_task_prior': independent[(scope,'gold_at_10',seed)]=number(value)
        else: check(f'final_path_table_consistent_seed_{seed}',number(value)==number(quality[seed if isinstance(seed,int) else 3][3]))
for row in table('5.2','| Split seed |'):
    independent[('ann_calibration','selected_ef_search',int(row[0]))]=number(row[1])
    independent[('ann_calibration','selected_recall_at_1000',int(row[0]))]=number(row[2])
for row in table('5.4','| Split seed | HNSW + rerank p50 |'):
    seed='mean' if row[0]=='Mean' else int(row[0])
    independent[('retrieval_cost','total_latency_p50_ms',seed)]=number(row[1])
    independent[('retrieval_cost','total_latency_p95_ms',seed)]=number(row[2])

text_patterns={
 ('retrieval_diagnostics','exact_pool_retention','mean'):r'Exact top-1,000 reranking retains ([0-9.]+)%',
 ('retrieval_diagnostics','ann_retention','mean'):r'HNSW retains ([0-9.]+)%',
 ('retrieval_diagnostics','overall_retention','mean'):r'for ([0-9.]+)% overall',
 ('retrieval_diagnostics','exact_pool_gold_coverage_at_1000','mean'):r'Dense top-1,000 contains the held-out gold for ([0-9.]+)%',
 ('retrieval_diagnostics','full_fused_top10_in_exact_pool','mean'):r'and ([0-9.]+)% of the exact full-pool fused top-ten members',
 ('retrieval_cost','hnsw_latency_p50_ms','mean'):r'Mean HNSW and reranking p50 components are ([0-9.]+) and',
 ('retrieval_cost','prior_rerank_latency_p50_ms','mean'):r'Mean HNSW and reranking p50 components are [0-9.]+ and ([0-9.]+) ms',
 ('index_cost','index_gib','sum'):r'the three total ([0-9.]+) GiB',
}
for key,pattern in text_patterns.items():
    independent[key]=float(re.search(pattern,sections['5.4']).group(1))
each=float(re.search(r'Each index occupies about ([0-9.]+) GiB',sections['5.4']).group(1))
build=re.search(r'were built in ([0-9.]+) / ([0-9.]+) / ([0-9.]+) seconds',sections['5.4'])
visible=re.search(r'only train\+validation edges: ([0-9,]+) / ([0-9,]+) / ([0-9,]+)',sections['5.3'])
for seed in (0,1,2):
    independent[('index_cost','index_gib',seed)]=each
    independent[('index_cost','build_seconds',seed)]=float(build.group(seed+1))
    independent[('task_prior','visible_prior_edges',seed)]=number(visible.group(seed+1))

# Structural/input numbers are checked in their cited source sections, with
# literal normalization solely for comma and numeric/word-zero formatting.
for r in old['records']:
    key=(r['scope'],r['name'],r['seed'])
    if key in independent:
        check('published_value:'+r['id'],r['value']==independent[key],{'stored':r['value'],'independent_source_parse':independent[key]})
    else:
        check('remaining_reference_is_frozen_input:'+r['id'],r['scope']=='frozen_input_audit')
        cited='\n'.join(sections[s.strip()] for s in r['source_section'].split(';'))
        canonical=cited.replace('{,}','').replace(',','')
        token=str(r['value'])
        match=bool(re.search(r'(?<![\d.])'+re.escape(token)+r'(?!\d|\.\d)',canonical))
        if r['name']=='model_snapshot_skipped_duplicates': match='zero skipped duplicates' in cited and r['value']==0
        check('published_input_value_in_cited_section:'+r['id'],match,{'literal':r['published_literal'],'section':r['source_section']})
    check('source_section_exists:'+r['id'],all(s.strip() in sections for s in r['source_section'].split(';')))
    check('historical_source_binding:'+r['id'],r['source_sha256']==sha(SOURCE) and r['source_path']==str(SOURCE))

metrics=inventory['metrics']; byid={m['id']:m for m in metrics}
required=set(inventory['required_fields'])
check('metric_ids_unique',len(byid)==len(metrics))
check('reference_ids_unique',len({r['id'] for r in old['records']})==len(old['records']))
check('all_required_fields_present',all(required<=set(m) for m in metrics))
check('all_new_values_null_and_pending',all(m['new_value'] is None and m['status']=='pending' for m in metrics))
check('no_fabricated_new_artifact_bindings',all(m['recomputed_from']==[] and m['artifact_hashes']=={} for m in metrics))
check('all_recompute_plans_nonempty',all(bool(m['recompute_from']) for m in metrics))
old_byid={r['id']:r for r in old['records']}
for m in metrics:
    r=old_byid.get(m['old_reference_id'])
    ok=(m['old_value']==r['value'] and m['old_source_section']==r['source_section'] and m['old_display_precision']==r['display_precision']) if r else (m['old_value'] is None and m['old_source']['status']=='not_reported')
    check('old_link:'+m['id'],ok)
    if m['seed'] in ('mean','min','max') and len(m['recompute_from'])==3 and all(x in byid for x in m['recompute_from']):
        check('summary_three_seed_inputs:'+m['id'],[byid[x]['seed'] for x in m['recompute_from']]==[0,1,2])
for name in ['exact_pool_retention','ann_retention','overall_retention']:
    s=byid['retrieval_diagnostics.'+name+'.mean']
    check('ratio_summary_is_mean_of_seed_ratios:'+name,'arithmetic mean' in s['aggregation'] and 'never ratio' in s['note'])
check('retention_divide_zero_rule','undefined' in protocol['reporting']['zero_denominator'])
check('timing_total_not_sum_of_percentiles','not sum' in protocol['evaluation']['timing']['cross_seed_summary'])
for path in ['hnsw1000_task_prior','exact1000_task_prior']:
    for seed in (0,1,2):
        rec=byid[f'{path}.median_gold_position_when_retrieved.{seed}']
        check(f'conditional_median_contract:{path}:{seed}',
              'COMPLETE' in rec['definition'] and 'conditioned' in rec['definition']
              and f'{path}.gold_retrieved_query_count.{seed}' in byid)
check('hnsw_coverage_no_misused_51_88',all(byid[f'hnsw1000_task_prior.actual_hnsw_gold_coverage_at_1000.{s}']['old_value'] is None for s in [0,1,2,'mean']))
check('final_gold10_mean_uses_high_precision',byid['hnsw1000_task_prior.gold_at_10.mean']['old_value']==float(raw.group(4)))
check('ef_first_passing_no_gold',protocol['evaluation']['ef_calibration']['selection']=='first passing grid value' and protocol['evaluation']['ef_calibration']['labels_used'] is False)

report={'stage':'A0.1','status':'PASS' if all(c['passed'] for c in checks) else 'FAIL',
 'review_method':'Independent authoritative table/paragraph parsing for every retrieval metric; section-specific source numeric/literal checks for all frozen-input references; full machine schema/linkage/aggregation review.',
 'reference_count':len(old['records']),'independently_parsed_retrieval_reference_count':len(independent),
 'inventory_count':len(metrics),'checks':checks,'failed':[c for c in checks if not c['passed']],
 'corrections_applied':[{'kind':'display_literal_only','detail':'ef displays use source thousands commas and skipped-duplicate literal is source word zero; numeric values unchanged'}],
 'mathematical_review':{'mean_of_seed_ratios':'required and registered','conditional_median':'complete reranked pool with own count; misses remain in quality denominators','rounded_old_values':'no reverse-engineered precision; primary gold@10 anchor retains ten published decimals','latency':'mean of per-seed quantiles; total quantile from total per-query elapsed array','unvisited_ef':'not_applicable after documented first pass; source rule preserved'},
 'artifact_hashes':{p.name:sha(p) for p in [SOURCE,OUT/'A0_PROTOCOL.json',OUT/'A0_OLD_REFERENCE.json',OUT/'A0_METRIC_INVENTORY.json']}}
target=AUDIT/'A0_REFERENCE_REVIEW.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'path':str(target),'status':report['status'],'checks':len(checks),'reference_count':len(old['records']),'inventory_count':len(metrics),'failed':report['failed'],'sha256':sha(target)}))
raise SystemExit(0 if report['status']=='PASS' else 1)
