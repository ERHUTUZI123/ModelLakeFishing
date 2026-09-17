"""A0.5 artifact checks; no ranking evaluation or parameter selection."""
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
ROOT=Path('/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing')
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
from scale1m.eval_y2 import TaskPrior,_load_candidates
seed=int(sys.argv[1]);OUT=Path(f'/u801/x98liu/model_lake/data1m/a0_20260912/exports/A0GD_full_s{seed}_e25')
GRAPH=OUT.parents[1]/'graph'
TASK=Path('/u801/x98liu/model_lake/data1m/rf/canon/dataset_nodes_merged.parquet')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest()
rep=json.loads((OUT/'EXPORT_MANIFEST.json').read_text())['stages']['embed']
assert rep['checkpoint_epoch']==24 and rep['split_seed']==seed
assert rep['graph_digest']=='acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db'
assert all(g['ok'] for g in rep['gates'])
for name,digest in rep['artifact_hashes'].items():assert sha(OUT/name)==digest,name
norm_checks={}
for name,rows in [('z_m',3016439),('z_m_eval',3016439),('z_d',18729),('z_d_eval',18729)]:
    a=np.load(OUT/(name+'.npy'),mmap_mode='r')
    assert a.shape==(rows,128) and a.dtype==np.float32
    delta=0.
    for start in range(0,rows,50000):
        block=a[start:start+50000];assert np.isfinite(block).all()
        delta=max(delta,float(np.abs(np.linalg.norm(block,axis=1)-1).max()))
    assert delta<1e-5,(name,delta)
    norm_checks[name]={'shape':list(a.shape),'dtype':str(a.dtype),'max_norm_error':delta,'finite':True}
mi=pd.read_parquet(OUT/'model_ids.parquet');di=pd.read_parquet(OUT/'dataset_ids.parquet')
for got,original in [(mi,'unique_model_id.parquet'),(di,'unique_dataset_id.parquet')]:
    pd.testing.assert_frame_equal(got,pd.read_parquet(GRAPH/original).sort_values('mappedID').reset_index(drop=True))
    assert np.array_equal(got.mappedID,np.arange(len(got)))
q=_load_candidates(str(OUT),str(TASK),seed)
identity=[json.loads(line) for line in (BASE/'A0_QUERY_IDENTITY.jsonl').read_text().splitlines()]
identity=[r for r in identity if r['seed']==seed]
assert sorted(q)==[r['query_mappedID'] for r in identity]
for r in identity:
    k=r['query_mappedID'];models,values=q[k]
    assert ah(models)==r['candidate_ids_sha256'] and ah(values)==r['oriented_values_float64_sha256']
    assert di.iloc[k]['dataset']==r['node'] and di.iloc[k]['root']==r['root']
scpath=OUT/f'prior_sidecar_s{seed}.npz';mp=OUT/f'prior_sidecar_s{seed}_meta.json'
meta=json.loads(mp.read_text());assert sha(scpath)==meta['sidecar_sha256']
assert meta['a0']==rep['a0'] or all(meta['a0'][k]==rep['a0'][k] for k in ['protocol','run_id','seed','graph_digest','checkpoint_sha256'])
prior=TaskPrior(str(scpath),str(mp),seed,q,di.root.to_numpy())
sc=prior.payload;em=sc['edge_model'];ed=sc['edge_dataset'];acc=sc['edge_acc']
audit=json.loads((BASE/'A0_INPUT_AUDIT.json').read_text())['splits'][seed]
assert ah(np.stack([em,ed]))==audit['prior_visible_pairs_ordered']['sha256_c_order_bytes']
tasks=pd.read_parquet(TASK).set_index('node').task.reindex(di.dataset)
assert tasks.notna().all()
norm=tasks.map(lambda t:re.sub(r'[\s_]+','-',str(t).strip().lower()))
code={v:i for i,v in enumerate(sorted(set(norm)))}
assert np.array_equal(sc['task_id'],norm.map(code).to_numpy())
assert len(code)>1 and len(code)==meta['n_tasks']
assert set(di.root.iloc[ed]).isdisjoint(di.root.iloc[list(q)])
counts={};sums={}
for t,m,a in zip(sc['task_id'][ed],em,acc):
    key=(int(t),int(m));counts[key]=counts.get(key,0)+1;sums[key]=sums.get(key,0.)+float(a)
max_delta=0.
for task,(models,values) in prior.by_task.items():
    expected=np.array([(sums[(task,int(m))]+2.5)/(counts[(task,int(m))]+5) for m in models])
    max_delta=max(max_delta,float(np.max(np.abs(values-expected))))
assert max_delta<1e-12,max_delta
zero_checks=0
for task in set(prior.task_id[list(q)]):
    query=next(k for k in q if prior.task_id[k]==task)
    evidence=prior.by_task.get(int(task),(np.array([],dtype=np.int64),None))[0]
    absent=np.setdiff1d(np.arange(3016439,dtype=np.int64),evidence,assume_unique=True)[:1000]
    assert len(absent)>0 and np.all(prior.values(query,absent)==0)
    zero_checks+=len(absent)
report={'status':'PASS','seed':seed,'checkpoint_sha256':rep['a0']['checkpoint_sha256'],'checkpoint_epoch':24,'graph_digest':rep['graph_digest'],'embedding_checks':norm_checks,'chunk_gates':rep['gates'],'queries':len(q),'query_ids_candidates_values_match_A01':True,'row_tables_match_graph':True,'visible_edges':meta['n_edges'],'visible_edge_order_matches_A01':True,'tasks':meta['n_tasks'],'scored_roots_excluded':True,'prior_formula_max_abs_error':max_delta,'no_evidence_zero_checks':zero_checks,'export_seconds':rep['seconds'],'prior_build_segments':meta['prior_build_segments'],'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file()}}
(BASE/f'A05_VALIDATION_s{seed}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','seed','queries','visible_edges','tasks','export_seconds']}),flush=True)
