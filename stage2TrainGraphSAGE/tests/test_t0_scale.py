import os
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
    TRAINED_ON, SparseMembership, high_performer_membership, topk_membership,
    pool_membership_by_root, global_positive_density, per_dataset_density,
    lineage_components, contrastive_loss, contrastive_loss_sampled,
)
from ModelLakeFishing.stage2TrainGraphSAGE.sampling import (
    LightLinkLoader, build_csr, batch_contrastive_masks, batch_positive_pairs,
)
from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE
from ModelLakeFishing.stage2TrainGraphSAGE.inference import chunked_forward
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval
from ModelLakeFishing.scale import global_metrics as GM
from ModelLakeFishing.scale.export_ours import tune_ef_for_recall


def toy_graph(n_m=400, n_d=60, seed=0, frozen_dim=16, ds_dim=12):
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["model"].x = torch.randn(n_m, frozen_dim, generator=g)
    data["model"].size_bucket_id = torch.randint(0, 5, (n_m,), generator=g)
    data["model"].family_id = torch.randint(0, 7, (n_m,), generator=g)
    data["dataset"].x = torch.randn(n_d, ds_dim, generator=g)

    n_e = n_m * 6
    src = torch.randint(0, n_m, (n_e,), generator=g)
    dst = torch.randint(0, n_d, (n_e,), generator=g)
    key = torch.unique(src * n_d + dst)
    src, dst = key // n_d, key % n_d
    acc = torch.rand(src.numel(), generator=g)
    data[TRAINED_ON].edge_index = torch.stack([src, dst])
    data[TRAINED_ON].edge_attr = acc
    data[("dataset", "rev_trained_on", "model")].edge_index = torch.stack([dst, src])
    data[("dataset", "rev_trained_on", "model")].edge_attr = acc

    ds_a = torch.randint(0, n_d, (n_d * 4,), generator=g)
    ds_b = torch.randint(0, n_d, (n_d * 4,), generator=g)
    data[("dataset", "similar_to", "dataset")].edge_index = torch.stack([ds_a, ds_b])
    data[("dataset", "similar_to", "dataset")].edge_attr = torch.rand(ds_a.numel(), generator=g)

    lin_a = torch.randint(0, n_m, (n_m // 4,), generator=g)
    lin_b = torch.randint(0, n_m, (n_m // 4,), generator=g)
    data[("model", "is_base_of", "model")].edge_index = torch.stack([lin_a, lin_b])
    data[("model", "is_base_of", "model")].edge_attr = torch.rand(lin_a.numel(), generator=g)
    data[("model", "rev_is_base_of", "model")].edge_index = torch.stack([lin_b, lin_a])
    data[("model", "rev_is_base_of", "model")].edge_attr = torch.rand(lin_a.numel(), generator=g)
    return data


def toy_model(data, seed=0):
    torch.manual_seed(seed)
    return HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=5, num_families=7,
        dataset_in_dim=data["dataset"].x.shape[1],
        hidden_channels=16, out_dim=8, num_layers=2).eval()


def test_build_csr_matches_edge_index():
    data = toy_graph()
    ei = data[TRAINED_ON].edge_index
    ptr, nbr = build_csr(ei, data["model"].num_nodes)
    for v in (0, 5, 17, data["model"].num_nodes - 1):
        got = sorted(nbr[ptr[v]:ptr[v + 1]].tolist())
        want = sorted(ei[1][ei[0] == v].tolist())
        assert got == want


def test_fanout_bounded():
    data = toy_graph()
    ei = data[TRAINED_ON].edge_index
    eli, elabel = ei[:, :128], data[TRAINED_ON].edge_attr[:128]
    B = 64
    R = 2 * len(data.edge_types)
    k = 2

    capped = LightLinkLoader(data, eli, elabel, num_hops=2, batch_size=B,
                             shuffle=False, fanout=[k, k], fanout_lineage=[k, k])
    full = LightLinkLoader(data, eli, elabel, num_hops=2, batch_size=B, shuffle=False)

    b_cap = next(iter(capped))
    b_full = next(iter(full))
    seeds = 2 * B
    bound = seeds * (1 + R * k + (R * k) ** 2)
    total = b_cap["model"].num_nodes + b_cap["dataset"].num_nodes
    assert total <= bound, f"{total} > analytic bound {bound}"
    assert b_cap["model"].num_nodes < b_full["model"].num_nodes
    assert b_full["model"].num_nodes > 0.9 * data["model"].num_nodes


def test_fanout_preserves_supervision():
    data = toy_graph()
    ei = data[TRAINED_ON].edge_index
    eli, elabel = ei[:, :128], data[TRAINED_ON].edge_attr[:128]
    loader = LightLinkLoader(data, eli, elabel, num_hops=2, batch_size=32,
                             shuffle=False, fanout=[2, 2])
    b = next(iter(loader))
    assert b[TRAINED_ON].edge_label_index.size(1) == 32
    assert torch.equal(b[TRAINED_ON].edge_label, elabel[:32])
    lm, ld = b[TRAINED_ON].edge_label_index
    assert torch.equal(b["model"].n_id[lm], eli[0][:32])
    assert torch.equal(b["dataset"].n_id[ld], eli[1][:32])


def test_fanout_lineage_gets_higher_cap():
    data = toy_graph()
    ei = data[TRAINED_ON].edge_index
    loader = LightLinkLoader(data, ei[:, :16], data[TRAINED_ON].edge_attr[:16],
                             num_hops=2, batch_size=16, fanout=[3, 3])
    assert loader.fanout_lineage == [6, 6]


def _dense_and_sparse(data, top_frac=0.1):
    ti = data[TRAINED_ON].edge_index
    ta = data[TRAINED_ON].edge_attr
    dense = topk_membership(data, top_frac=top_frac, trained_on_index=ti,
                            trained_on_attr=ta)
    sparse = topk_membership(data, top_frac=top_frac, trained_on_index=ti,
                             trained_on_attr=ta, sparse=True)
    return dense, sparse, ti


def test_sparse_M_equivalence():
    data = toy_graph()
    dense, sparse, _ = _dense_and_sparse(data)
    assert torch.equal(sparse.to_dense(), dense)
    assert sparse.nnz == int(dense.sum())
    assert torch.equal(sparse.sum(0), dense.sum(0))
    for d in (0, 3, 11, dense.shape[1] - 1):
        assert torch.equal(sparse[:, d], dense[:, d])
        assert torch.equal(torch.sort(sparse.col_ids(d)).values,
                           (dense[:, d] > 0).nonzero().flatten())
    n_id = torch.tensor([0, 5, 9, 17, 100, 399])
    assert torch.equal(sparse[n_id], dense[n_id])
    assert torch.equal(sparse.rows_dense(n_id), dense[n_id])
    cand = torch.tensor([0, 1, 2, 3])
    assert torch.equal(sparse[cand, 2], dense[cand, 2])
    assert float(sparse[7, 2]) == float(dense[7, 2])


def test_sparse_M_dedupes_repeated_pairs():
    m = SparseMembership(torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]), (4, 3))
    assert m.nnz == 2
    assert m.to_dense().max().item() == 1.0


def test_sparse_M_memory_is_a_real_saving():
    data = toy_graph()
    dense, sparse, _ = _dense_and_sparse(data)
    dense_bytes = dense.numel() * dense.element_size()
    sparse_bytes = sum(t.numel() * t.element_size() for t in
                       (sparse.col_rows, sparse.col_ptr, sparse.row_cols, sparse.row_ptr))
    assert sparse_bytes < dense_bytes / 4


def test_pool_membership_by_root_equivalence():
    data = toy_graph()
    dense, sparse, _ = _dense_and_sparse(data)
    n_d = dense.shape[1]
    root_ids = torch.arange(n_d) // 4
    got = pool_membership_by_root(dense, root_ids)
    ref = torch.zeros_like(dense)
    for r in torch.unique(root_ids).tolist():
        cols = (root_ids == r).nonzero().flatten()
        ref[:, cols] = (dense[:, cols].sum(1, keepdim=True) > 0).float()
    assert torch.equal(got, ref)
    got_sparse = pool_membership_by_root(sparse, root_ids)
    assert torch.equal(got_sparse.to_dense(), ref)


def test_pool_membership_singleton_roots_unchanged():
    data = toy_graph()
    dense, sparse, _ = _dense_and_sparse(data)
    root_ids = torch.arange(dense.shape[1])
    assert torch.equal(pool_membership_by_root(dense, root_ids), dense)
    assert torch.equal(pool_membership_by_root(sparse, root_ids).to_dense(), dense)


def test_density_diagnostics_agree():
    data = toy_graph()
    dense, sparse, ti = _dense_and_sparse(data)
    assert per_dataset_density(ti, dense) == pytest.approx(per_dataset_density(ti, sparse))
    a = global_positive_density(dense, max_dense=0, n_sample=5000,
                                generator=torch.Generator().manual_seed(3))
    b = global_positive_density(sparse, max_dense=0, n_sample=5000,
                                generator=torch.Generator().manual_seed(3))
    assert a == pytest.approx(b)
    c = global_positive_density(sparse, max_dense=0, n_sample=5000, pair_chunk=17,
                                generator=torch.Generator().manual_seed(3))
    assert a == pytest.approx(c)


def _batch_and_masks(sparse_M=False):
    data = toy_graph()
    ti, ta = data[TRAINED_ON].edge_index, data[TRAINED_ON].edge_attr
    M = topk_membership(data, top_frac=0.2, trained_on_index=ti, trained_on_attr=ta,
                        sparse=sparse_M)
    comp = lineage_components(data, data["model"].num_nodes)
    loader = LightLinkLoader(data, ti[:, :64], ta[:64], num_hops=1, batch_size=64,
                             shuffle=False, fanout=[4, 4])
    batch = next(iter(loader))
    return batch, M, comp


def test_batch_positive_pairs_matches_dense_mask():
    for sparse_M in (False, True):
        batch, M, comp = _batch_and_masks(sparse_M)
        pos, _hub = batch_contrastive_masks(batch, M, comp)
        ai, bi = batch_positive_pairs(batch, M)
        got = torch.zeros_like(pos)
        got[ai, bi] = True
        assert torch.equal(got, pos), f"sparse_M={sparse_M}"


def test_contrastive_sampled_equals_dense_when_not_sampling():
    batch, M, comp = _batch_and_masks()
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(batch["model"].num_nodes, 8), dim=-1)
    pos, hub = batch_contrastive_masks(batch, M, comp)
    ref = contrastive_loss(z, pos, hub)
    got = contrastive_loss_sampled(z, batch_positive_pairs(batch, M),
                                   comp[batch["model"].n_id], n_neg=None)
    assert float(got) == pytest.approx(float(ref), abs=1e-5)


def test_contrastive_sampled_hard_negative_weight_matters():
    batch, M, comp = _batch_and_masks()
    torch.manual_seed(1)
    z = torch.nn.functional.normalize(torch.randn(batch["model"].num_nodes, 8), dim=-1)
    pairs = batch_positive_pairs(batch, M)
    cb = comp[batch["model"].n_id]
    a = contrastive_loss_sampled(z, pairs, cb, n_neg=None, hard_neg_weight=1.0)
    b = contrastive_loss_sampled(z, pairs, cb, n_neg=None, hard_neg_weight=8.0)
    assert float(b) > float(a)


def test_contrastive_sampled_is_bounded_and_differentiable():
    batch, M, comp = _batch_and_masks()
    torch.manual_seed(2)
    z = torch.nn.functional.normalize(
        torch.randn(batch["model"].num_nodes, 8, requires_grad=True), dim=-1)
    loss = contrastive_loss_sampled(z, batch_positive_pairs(batch, M),
                                    comp[batch["model"].n_id], n_neg=8)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is None or torch.isfinite(z.grad).all()


def test_contrastive_sampled_approaches_exact_as_n_neg_grows():
    batch, M, comp = _batch_and_masks()
    torch.manual_seed(3)
    B = batch["model"].num_nodes
    z = torch.nn.functional.normalize(torch.randn(B, 8), dim=-1)
    pairs, cb = batch_positive_pairs(batch, M), comp[batch["model"].n_id]
    exact = float(contrastive_loss_sampled(z, pairs, cb, n_neg=None))
    err = []
    for n in (4, 64, 1024):
        vals = [float(contrastive_loss_sampled(z, pairs, cb, n_neg=n)) for _ in range(5)]
        err.append(abs(float(np.mean(vals)) - exact))
    assert err[-1] < err[0]


def test_batch_positive_pairs_cap():
    batch, M, comp = _batch_and_masks()
    full = batch_positive_pairs(batch, M)[0].numel()
    capped = batch_positive_pairs(batch, M, max_per_dataset=1)[0].numel()
    assert capped <= full


@pytest.mark.parametrize("chunk", [1, 7, 128, 10_000])
def test_chunked_forward_matches_full(chunk):
    data = toy_graph(n_m=200, n_d=40)
    model = toy_model(data)
    with torch.no_grad():
        ref = model(data.clone())
    got = chunked_forward(model, data, chunk_size=chunk, device="cpu")
    for nt in ("model", "dataset"):
        assert got[nt].shape == ref[nt].shape
        delta = (got[nt] - ref[nt]).abs().max().item()
        assert delta < 1e-5, f"{nt} chunk={chunk} max|delta|={delta}"


def test_chunked_forward_row_order_is_by_global_id():
    data = toy_graph(n_m=200, n_d=40)
    model = toy_model(data)
    with torch.no_grad():
        ref = model(data.clone())["model"]
    got = chunked_forward(model, data, chunk_size=37)["model"]
    assert torch.allclose(got, ref, atol=1e-5)
    for i in (0, 36, 37, 199):
        assert torch.allclose(got[i], ref[i], atol=1e-5)
        j = (i + 1) % 200
        assert not torch.allclose(got[i], ref[j], atol=1e-5)


def test_chunked_forward_one_layer():
    data = toy_graph(n_m=150, n_d=30)
    torch.manual_seed(0)
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=5, num_families=7,
        dataset_in_dim=data["dataset"].x.shape[1],
        hidden_channels=16, out_dim=8, num_layers=1).eval()
    with torch.no_grad():
        ref = model(data.clone())
    got = chunked_forward(model, data, chunk_size=32)
    assert (got["model"] - ref["model"]).abs().max().item() < 1e-5


def _cands(rng, N, D, n_q=40):
    cands = {}
    for d in range(n_q):
        n = int(rng.integers(3, 30))
        idx = rng.choice(N, size=n, replace=False)
        acc = rng.random(n).astype(np.float32)
        cands[d] = (idx, acc)
    return cands


def test_streaming_metrics_equal_dense():
    rng = np.random.default_rng(0)
    N, D, dim = 800, 50, 16
    z_m = rng.standard_normal((N, dim)).astype(np.float32)
    z_d = rng.standard_normal((D, dim)).astype(np.float32)
    cands = _cands(rng, N, D)
    roots = {d: d // 3 for d in cands}
    agg_ref, per_ref = GM.from_embeddings(z_m, z_d, cands, roots)
    for mc, qc in ((50_000, 64), (97, 7), (13, 1)):
        agg, per = GM.from_embeddings_streaming(z_m, z_d, cands, roots,
                                                model_chunk=mc, query_chunk=qc)
        for d in per_ref:
            assert per[d] == per_ref[d], f"query {d} chunk=({mc},{qc})"
        assert agg == pytest.approx(agg_ref)


def test_streaming_matches_five_metric_eval():
    rng = np.random.default_rng(1)
    N, D, dim = 600, 40, 12
    z_m = rng.standard_normal((N, dim)).astype(np.float32)
    z_d = rng.standard_normal((D, dim)).astype(np.float32)
    cands = _cands(rng, N, D)
    per_five = five_metric_eval({"model": torch.from_numpy(z_m),
                                 "dataset": torch.from_numpy(z_d)}, cands)
    _agg, per_stream = GM.from_embeddings_streaming(z_m, z_d, cands)
    for d in per_five:
        assert per_five[d]["gold_rank"] == per_stream[d]["gold_rank"]
        assert per_five[d]["gold_gap_rank"] == per_stream[d]["gap_rank"]


def test_five_metric_expect_n_fires():
    rng = np.random.default_rng(2)
    N, D, dim = 100, 10, 8
    z_m = rng.standard_normal((N, dim)).astype(np.float32)
    z_d = rng.standard_normal((D, dim)).astype(np.float32)
    cands = _cands(rng, N, D, n_q=5)
    z = {"model": torch.from_numpy(z_m), "dataset": torch.from_numpy(z_d)}
    five_metric_eval(z, cands, expect_n=N)
    with pytest.raises(AssertionError):
        five_metric_eval(z, cands, expect_n=N + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_five_metric_gpu_cpu_rank_parity():
    rng = np.random.default_rng(3)
    N, D, dim = 700, 30, 16
    z_m = rng.standard_normal((N, dim)).astype(np.float32)
    z_d = rng.standard_normal((D, dim)).astype(np.float32)
    cands = _cands(rng, N, D, n_q=20)
    z = {"model": torch.from_numpy(z_m), "dataset": torch.from_numpy(z_d)}
    a = five_metric_eval(z, cands)
    b = five_metric_eval(z, cands, device="cuda")
    for d in a:
        assert a[d]["gold_rank"] == b[d]["gold_rank"]


def test_train_loop_with_all_switches():
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import PerfScorer, accuracy_lookup
    from ModelLakeFishing.stage2TrainGraphSAGE.train import train

    data = toy_graph(n_m=300, n_d=40)
    model = toy_model(data)
    model.train()
    scorer = PerfScorer(dim=8, mode="dot")
    lookup = accuracy_lookup(data)
    ti = data[TRAINED_ON].edge_index
    ta = data[TRAINED_ON].edge_attr
    eli, target = ti[:, :256], ta[:256]
    M = topk_membership(data, top_frac=0.2, trained_on_index=ti, trained_on_attr=ta,
                        sparse=True)
    comp = lineage_components(data, data["model"].num_nodes)

    hist, _ = train(model, scorer, data, eli, target, M, comp,
                    epochs=6, batch_size=64, lambda_rank=1.0, lambda_contrast=1.0,
                    lambda_mse=0.0, lambda_uniform=0.0,
                    fanout=True, num_neighbors=(4, 4), contrast_n_neg=16,
                    contrast_max_pos_per_dataset=32)
    assert all(np.isfinite(h["total"]) for h in hist)
    assert hist[-1]["total"] < hist[0]["total"]
    assert model.model_encoder.size_embedding.weight.grad.abs().sum() > 0
    assert model.model_encoder.family_embedding.weight.grad.abs().sum() > 0
    assert data["model"].x.grad is None


def test_train_loop_switches_off_is_the_legacy_path():
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import PerfScorer
    from ModelLakeFishing.stage2TrainGraphSAGE.train import train

    data = toy_graph(n_m=200, n_d=30)
    ti, ta = data[TRAINED_ON].edge_index, data[TRAINED_ON].edge_attr
    eli, target = ti[:, :128], ta[:128]
    M = topk_membership(data, top_frac=0.2, trained_on_index=ti, trained_on_attr=ta)
    comp = lineage_components(data, data["model"].num_nodes)

    def run(**kw):
        torch.manual_seed(7)
        m = toy_model(data)
        m.train()
        s = PerfScorer(dim=8, mode="dot")
        torch.manual_seed(11)
        h, _ = train(m, s, data, eli, target, M, comp, epochs=3, batch_size=64, **kw)
        return [x["total"] for x in h]

    assert run() == run(fanout=False, contrast_n_neg=None)


def test_tune_ef_for_recall():
    hnswlib = pytest.importorskip("hnswlib")
    rng = np.random.default_rng(0)
    N, D, dim = 4000, 30, 32
    zm = rng.standard_normal((N, dim)).astype(np.float32)
    zm /= np.linalg.norm(zm, axis=1, keepdims=True)
    zd = rng.standard_normal((D, dim)).astype(np.float32)
    zd /= np.linalg.norm(zd, axis=1, keepdims=True)
    idx = hnswlib.Index(space="ip", dim=dim)
    idx.init_index(max_elements=N, ef_construction=100, M=8)
    idx.add_items(zm, np.arange(N))
    qd = list(range(D))
    ef, rec, trace = tune_ef_for_recall(idx, zm, zd, qd, target=0.99, K=50)
    assert rec >= 0.99
    assert ef >= 50 and len(trace) >= 1
    failing = [t["ef"] for t in trace if t["recall"] < 0.99]
    assert all(f < ef for f in failing)
