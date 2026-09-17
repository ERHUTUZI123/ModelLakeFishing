"""A0.3: held-out labels cannot change GD training or held-out embeddings."""
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import pytest

from scale1m.build_graph_rf import dataset_stats, mask_performance_features
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from ModelLakeFishing.stage2TrainGraphSAGE import ablation as A
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
from ModelLakeFishing.stage2TrainGraphSAGE.inference import chunked_forward
from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, REV_TRAINED_ON, accuracy_lookup, perf_supervision
from ModelLakeFishing.stage2TrainGraphSAGE.tests.test_t0_scale import toy_graph


@pytest.fixture
def deterministic_cpu_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_fixed_test_root_weights_and_eligibility_leave_real_gd_update_unchanged(monkeypatch,deterministic_cpu_threads):
    cfg = json.loads((Path(__file__).resolve().parents[2] / 'docs/1M/A0_runs/A0_PROTOCOL.json').read_text(encoding='utf-8'))['required_config_audit']['resolved_config']
    raw = toy_graph(n_m=120, n_d=24, frozen_dim=448, ds_dim=458)
    roots = [f'root{i//2}' for i in range(24)]
    nodes = pd.DataFrame({'dataset':roots, 'gold_eligible':[True]*24})
    clean, _ = dataset_stats(nodes, None)
    raw['dataset'].x[:,448:] = torch.from_numpy(clean)
    for field in ('task_type_id','n_class_bucket_id','arity_id'):
        raw['dataset'][field] = torch.zeros(24,dtype=torch.long)
    base = apply_similar_to_mode(raw, cfg['similar_to_mode'], k=cfg['similar_to_k'])
    split0 = make_root_aware_splits(base, roots, split_seed=0)
    labels = split0[2][TRAINED_ON]
    query_ids = labels.edge_label_index[1,labels.edge_label > 0].unique()
    test_roots = {roots[i] for i in query_ids.tolist()}
    changed = base.clone()
    affected = torch.tensor([roots[d] in test_roots for d in base[TRAINED_ON].edge_index[1]])
    changed[TRAINED_ON].edge_attr[affected] = 1 - changed[TRAINED_ON].edge_attr[affected]
    changed[REV_TRAINED_ON].edge_attr = changed[TRAINED_ON].edge_attr.clone()
    nodes2 = nodes.copy(); nodes2.loc[[r in test_roots for r in roots],'gold_eligible'] = False
    stats2, _ = dataset_stats(nodes2, pd.DataFrame({'weight':changed[TRAINED_ON].edge_attr.numpy()}))
    assert np.array_equal(clean,stats2)
    noisy = changed['dataset'].x.numpy().copy()
    noisy[:,448:454] = 123; noisy[:,455] = 1
    changed['dataset'].x = torch.from_numpy(mask_performance_features(noisy))
    assert torch.equal(base['dataset'].x,changed['dataset'].x)
    split1 = make_root_aware_splits(changed,roots,split_seed=0)
    for left,right in zip(split0,split1):
        for relation in (TRAINED_ON,REV_TRAINED_ON):
            assert torch.equal(left[relation].edge_index,right[relation].edge_index)
            assert torch.equal(left[relation].edge_attr,right[relation].edge_attr)
        assert torch.equal(left[TRAINED_ON].edge_label_index,right[TRAINED_ON].edge_label_index)
        assert torch.equal(left[REV_TRAINED_ON].edge_index,left[TRAINED_ON].edge_index.flip(0))
    sides=[]
    for split in split0:
        store=split[TRAINED_ON]
        sides.append({roots[d] for d in store.edge_label_index[1,store.edge_label>0].tolist()})
    assert not (sides[0]&sides[1] or sides[0]&sides[2] or sides[1]&sides[2])
    assert all(roots[d] not in test_roots for d in split0[2][TRAINED_ON].edge_index[1].tolist())
    look0,look1=accuracy_lookup(base),accuracy_lookup(changed)
    assert look0 != look1
    for a,b in zip(perf_supervision(split0[0][TRAINED_ON],look0),perf_supervision(split1[0][TRAINED_ON],look1)):
        assert torch.equal(a,b)

    actual_train=A.train
    calls=[]
    def observe(*args,**kwargs):
        calls.append(tuple(t.detach().cpu().clone() for t in kwargs['global_ctx']['lake']))
        return actual_train(*args,**kwargs)
    monkeypatch.setattr(A,'train',observe)
    def run(data,split):
        random.seed(0)
        history=[]
        local=dict(cfg,on_epoch_end=lambda epoch,metrics,ctx: history.append(dict(metrics)))
        _,_,_,model,scorer=A.train_eval_one(data,{'num_size_buckets':5,'num_families':7},
            {'num_task_types':1,'n_class_buckets':1,'num_arities':1},local,split,
            init_seed=0,epochs=1,device='cpu',smoke_only=True)
        model.eval()
        with torch.no_grad():
            z=model(split[2])
            chunk=chunked_forward(model,split[2],chunk_size=17,device='cpu')
        for key,rows in (('model',120),('dataset',24)):
            assert z[key].shape==(rows,128) and torch.isfinite(z[key]).all()
            assert torch.allclose(z[key].norm(dim=1),torch.ones(rows),atol=1e-5,rtol=0)
            assert (z[key]-chunk[key]).abs().max()<1e-5
        return history,model.state_dict(),scorer.state_dict(),z
    left,right=run(base,split0),run(changed,split1)
    assert left[0]==right[0] and all(np.isfinite(v) for v in left[0][0].values() if isinstance(v,(float,int)))
    assert len(calls)==2 and all(torch.equal(a,b) for a,b in zip(*calls))
    for a,b in zip(left[1:],right[1:]):
        assert a.keys()==b.keys()
        differences={k:float((a[k]-b[k]).abs().max()) for k in a if not torch.equal(a[k],b[k])}
        assert not differences,differences
