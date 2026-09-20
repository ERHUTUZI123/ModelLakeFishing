import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from scale1m.build_graph_rf import A0_ZERO_COLUMNS, dataset_stats, mask_performance_features
from scale1m.checkpoint import graph_digest, sha256_of
from scale1m.graph_store import save_sharded
from scale1m.prepare_a0_graph import prepare_graph, verify_prepared
from scale1m.a0_graph_validation import verify_a0_graph


def tiny_source(path):
    rng = np.random.default_rng(21)
    data = HeteroData()
    data["model"].x = torch.from_numpy(rng.normal(size=(7,448)).astype(np.float32))
    data["model"].node_id = torch.arange(7)
    data["model"].family_id = torch.zeros(7,dtype=torch.int64)
    data["model"].size_bucket_id = torch.zeros(7,dtype=torch.int64)
    x = rng.normal(size=(3,458)).astype(np.float32)
    x[:,454] = np.log1p([2,2,1])
    x[:,456:] = 0
    data["dataset"].x = torch.from_numpy(x)
    data["dataset"].node_id = torch.arange(3)
    forward = ("model","trained_on","dataset")
    reverse = ("dataset","rev_trained_on","model")
    data[forward].edge_index = torch.tensor([[0,1,2,3],[0,0,1,2]])
    data[forward].edge_attr = torch.tensor([0.5,0.7,0.2,0.9])
    data[reverse].edge_index = data[forward].edge_index.flip(0)
    data[reverse].edge_attr = data[forward].edge_attr.clone()
    ckpt = {"data": data, "xm0_meta": {"num_families":1,"num_size_buckets":1},
            "xd0_meta": {"view_dims":{"e_name":64,"e_card":384,"e_stats":10}},
            "unique_model_id":pd.DataFrame({"model":[f"m/{i}" for i in range(7)],"mappedID":range(7)}),
            "unique_dataset_id":pd.DataFrame({"dataset":["r/a\tt1","r/b\tt2","s/c\tt1"],
                                               "mappedID":range(3),"root":["r","r","s"]})}
    save_sharded(ckpt,str(path))
    return graph_digest(str(path))


def test_stats_ignore_every_performance_and_eligibility_value():
    nodes=pd.DataFrame({"dataset":["r/a","r/b","s/c"],"gold_eligible":[True,False,True]})
    edges=pd.DataFrame({"node":["x","y"],"weight":[0.0,1.0]})
    expected=np.zeros((3,10),np.float32)
    expected[:,6]=np.log1p([2,2,1])
    a, roots=dataset_stats(nodes,edges)
    perturbed=nodes.copy(); perturbed["gold_eligible"]=[False,True,False]
    b,_=dataset_stats(perturbed,pd.DataFrame({"weight":[np.nan,np.inf,-9.]}))
    c,_=dataset_stats(nodes.drop(columns="gold_eligible"),None)
    assert np.array_equal(a,expected) and np.array_equal(a,b) and np.array_equal(a,c)
    assert roots==["r","r","s"]


def test_mask_copies_and_preserves_every_other_bit(tmp_path):
    path=tmp_path/'x.npy'
    original=np.random.default_rng(4).normal(size=(6,458)).astype(np.float32)
    np.save(path,original)
    before=sha256_of(path)
    frozen=np.load(path,mmap_mode='r')
    clean=mask_performance_features(frozen)
    unchanged=np.setdiff1d(np.arange(458),A0_ZERO_COLUMNS)
    assert np.array_equal(clean[:,unchanged].view(np.uint32),original[:,unchanged].view(np.uint32))
    assert np.all(clean[:,A0_ZERO_COLUMNS]==0)
    assert sha256_of(path)==before
    clean[0,0]=123
    assert frozen[0,0]!=123


@pytest.mark.parametrize('bad',[np.ones((2,457),np.float32),np.ones((2,458),np.float64),np.full((2,458),np.nan,np.float32)])
def test_mask_rejects_bad_shape_precision_or_nonfinite(bad):
    with pytest.raises(ValueError): mask_performance_features(bad)


def test_preparation_changes_only_seven_columns_and_binds_complete_files(tmp_path):
    source,out=tmp_path/'source',tmp_path/'out'
    digest=tiny_source(source)
    original={p.name:sha256_of(p) for p in source.iterdir()}
    report=prepare_graph(source,out,expected_source_digest=digest,expected_shape=(7,3))
    assert report['status']=='PASS' and report['new_graph_digest']!=digest
    assert report['actual_changed_columns']==list(A0_ZERO_COLUMNS)
    assert report['load_sharded_verify_sha256'] and report['other_graph_inputs_byte_identical']
    assert {p.name:sha256_of(p) for p in source.iterdir()}==original
    meta=json.loads((out/'meta.json').read_text())
    assert set(meta['files'])=={p.name for p in out.iterdir()}-{'meta.json'}
    assert all(sha256_of(out/name)==value for name,value in meta['files'].items())
    assert verify_prepared(source,out,expected_source_digest=digest,expected_shape=(7,3))['status']=='PASS'
    assert verify_a0_graph(out,require_frozen_source=False)['graph_sha256']==report['new_graph_digest']
    with pytest.raises(FileExistsError): prepare_graph(source,out,expected_source_digest=digest,expected_shape=(7,3))


@pytest.mark.parametrize('mode',['feature_without_rehash','feature_with_rehash','edge_with_rehash','unbound_file','wrong_root_count','wrong_source_digest','encoder_metadata'])
def test_corruption_and_unauthorized_changes_are_blocked(tmp_path,mode):
    source,out=tmp_path/'source',tmp_path/'out'
    digest=tiny_source(source)
    if mode=='wrong_source_digest':
        with pytest.raises(ValueError,match='frozen graph digest'):
            prepare_graph(source,out,expected_source_digest='0'*64,expected_shape=(7,3))
        assert not out.exists()
        return
    if mode=='wrong_root_count':
        x=np.load(source/'x_dataset.npy'); x[0,454]=8
        np.save(source/'x_dataset.npy',x)
        meta=json.loads((source/'meta.json').read_text()); meta['files']['x_dataset.npy']=sha256_of(source/'x_dataset.npy')
        (source/'meta.json').write_text(json.dumps(meta))
        with pytest.raises(ValueError,match='root-count'):
            prepare_graph(source,out,expected_source_digest=graph_digest(str(source)),expected_shape=(7,3))
        return
    prepare_graph(source,out,expected_source_digest=digest,expected_shape=(7,3))
    meta=json.loads((out/'meta.json').read_text())
    if mode=='unbound_file':
        (out/'extra.json').write_text('{}')
    elif mode=='encoder_metadata':
        meta['xm0_meta']['num_families']=99
    elif mode.startswith('feature'):
        x=np.load(out/'x_dataset.npy'); x[0,64]+=1
        np.save(out/'x_dataset.npy',x)
        if mode=='feature_with_rehash': meta['files']['x_dataset.npy']=sha256_of(out/'x_dataset.npy')
    else:
        with (out/'edges.npz').open('ab') as f: f.write(b'altered')
        meta['files']['edges.npz']=sha256_of(out/'edges.npz')
    (out/'meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError): verify_prepared(source,out,expected_source_digest=digest,expected_shape=(7,3))
    with pytest.raises(ValueError): verify_a0_graph(out,require_frozen_source=False)


def test_export_rejects_renamed_smoke_checkpoint_before_inference(tmp_path,monkeypatch):
    from scale1m import export_rf
    monkeypatch.setattr(export_rf.CK,'load',lambda _: {'extra':{'smoke_only':True},'binding':{}})
    with pytest.raises(export_rf.CK.IncompatibleCheckpoint,match='smoke'):
        export_rf.bind_or_die(str(tmp_path/'renamed.pt'),str(tmp_path/'irrelevant_graph'))


def test_source_and_output_cannot_overlap(tmp_path):
    source=tmp_path/'source'; digest=tiny_source(source)
    for out in (source,source/'nested',tmp_path):
        with pytest.raises(ValueError,match='separate'):
            prepare_graph(source,out,expected_source_digest=digest,expected_shape=(7,3))
