"""A0.6 delivery integrity; independent full metric recomputation is A0.7."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=Path('/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing')
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
from scale1m.a0_evaluation import verify_artifacts
OUT=BASE.parent/'data1m/a0_20260912/metrics'
report=json.loads((OUT/'A0_EVALUATION_REPORT.json').read_text())
manifest=json.loads((OUT/'A0_EVALUATION_MANIFEST.json').read_text())
assert report['computation_complete'] and report['stage']==manifest['stage']=='complete'
assert report['primary_path']=='G_hnsw1000_task' and report['K']==1000 and report['return_k']==10
assert all(report['correctness_gates'].values()) and report['query_observations']==4122
verify_artifacts(OUT,manifest)
checks=[]
for seed,n in enumerate([1476,1101,1545]):
    entry=manifest['seeds'][str(seed)];r=report['hnsw'][str(seed)]
    assert entry['index_build']['build_seconds']>0
    assert entry['selected_ef'] in [1000,1500,2000,3000,5000]
    with np.load(OUT/entry['exact'],allow_pickle=False) as exact,np.load(OUT/entry['hnsw'],allow_pickle=False) as hnsw:
        assert np.array_equal(exact['query'],hnsw['query']) and len(hnsw['query'])==n
        assert hnsw['model'].shape==(n,1000)
        assert all(len(np.unique(row))==1000 for row in hnsw['model'])
        for name in ['score','prior','fused']:
            assert np.isfinite(hnsw[name]).all()
        assert np.array_equal(hnsw['total_ns'],hnsw['hnsw_ns']+hnsw['rerank_ns'])
        assert (hnsw['total_ns']>0).all()
    checks.append({'seed':seed,'queries':n,'build_seconds':r['build_seconds'],'ef_search':r['ef_search'],'recall@1000':r['recall@1000'],'recall_gate':r['gate_recall@1000'],'final_quality':r['rows']['G_hnsw1000_task'],'latency_ms':r['latency_ms']})
validation={'status':'PASS','computation_complete':True,'effectiveness_gates':report['effectiveness_gates'],'ann_fidelity_status':report['ann_fidelity_status'],'seed_checks':checks,'independent_all_metric_recomputation':'A0.7 pending','metrics_path':str(OUT),'artifact_count':len(manifest['artifacts'])}
(BASE/'A06_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation),flush=True)
