import json
import os

import numpy as np
import pytest
import torch

from scale1m import checkpoint as CK


class Tiny(torch.nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


def _fixture(tmp_path):
    torch.manual_seed(0)
    model, scorer = Tiny(), Tiny()
    opt = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()),
                           lr=1e-2)
    binding = {"family_vocab_sha256": "abc", "num_families": 1896,
               "num_size_buckets": 15,
               "size_bucket_version": CK.size_bucket_version(),
               "name_seed": 42, "name_dim": 64, "desc_dim": 384,
               "encoder_name": "all-MiniLM-L6-v2", "graph_sha256": "deadbeef",
               "split_seed": 0}
    return model, scorer, opt, binding


def _step(model, scorer, opt, n=1):
    for _ in range(n):
        opt.zero_grad()
        x = torch.randn(2, 4)
        (model(x).sum() + scorer(x).pow(2).sum()).backward()
        opt.step()


def test_save_writes_numbered_file_and_repoints_last(tmp_path):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    p = CK.save(d, epoch=4, global_step=5, model=model, scorer=scorer, opt=opt,
                history=[{"total": 1.0}], cfg={"lr": 0.01}, binding=binding)
    assert os.path.basename(p) == "checkpoint_epoch_4.pt"
    assert os.path.exists(os.path.join(d, CK.LAST))
    ck = CK.load(os.path.join(d, CK.LAST))
    assert ck["epoch"] == 4 and ck["global_step"] == 5
    assert ck["binding"]["num_families"] == 1896
    for k in ("model", "scorer", "opt", "rng", "cfg", "history"):
        assert k in ck


def test_save_leaves_no_tmp_files_behind(tmp_path):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    CK.save(d, epoch=0, global_step=1, model=model, scorer=scorer, opt=opt,
            history=[], cfg={}, binding=binding)
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]


def test_graph_digest_matches_sha256_for_a_single_file(tmp_path):
    graph = tmp_path / "graph.pt"
    graph.write_bytes(b"single-file graph")
    assert CK.graph_digest(str(graph)) == CK.sha256_of(str(graph))


def test_graph_digest_is_stable_and_tracks_shard_hashes(tmp_path):
    graph = tmp_path / "graph"
    graph.mkdir()
    meta = graph / "meta.json"
    meta.write_text(json.dumps({"files": {"nodes.npz": "a", "edges.npz": "b"}}),
                    encoding="utf-8")
    first = CK.graph_digest(str(graph))
    assert CK.graph_digest(str(graph)) == first
    meta.write_text(json.dumps({"files": {"nodes.npz": "changed", "edges.npz": "b"}}),
                    encoding="utf-8")
    assert CK.graph_digest(str(graph)) != first


def test_prune_keeps_best_last_and_history(tmp_path):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    for e in range(6):
        p = CK.save(d, epoch=e, global_step=e + 1, model=model, scorer=scorer,
                    opt=opt, history=[], cfg={}, binding=binding, keep=2)
    CK.save_best(d, p)
    numbered = sorted(f for f in os.listdir(d) if f.startswith(CK.STEM))
    assert numbered == ["checkpoint_epoch_4.pt", "checkpoint_epoch_5.pt"]
    assert os.path.exists(os.path.join(d, CK.LAST))
    assert os.path.exists(os.path.join(d, CK.BEST))


def test_resume_order_prefers_explicit_then_last_then_fresh(tmp_path):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    assert CK.resolve_resume(None, d) == (None, "fresh")

    p = CK.save(d, epoch=0, global_step=1, model=model, scorer=scorer, opt=opt,
                history=[], cfg={}, binding=binding)
    assert CK.resolve_resume(None, d) == (os.path.join(d, CK.LAST), "last.pt")
    assert CK.resolve_resume(p, d) == (p, "explicit")


def test_resume_with_a_missing_explicit_path_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        CK.resolve_resume(str(tmp_path / "nope.pt"), str(tmp_path))


def test_validate_accepts_a_matching_binding(tmp_path):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    CK.save(d, epoch=0, global_step=1, model=model, scorer=scorer, opt=opt,
            history=[], cfg={}, binding=binding)
    assert CK.validate(CK.load(os.path.join(d, CK.LAST)), binding) == []


@pytest.mark.parametrize("key,value", [
    ("family_vocab_sha256", "different"),
    ("num_families", 341),
    ("graph_sha256", "another-graph"),
    ("split_seed", 1),
    ("size_bucket_version", "min5.0_max12.0_w1.00_n8"),
])
def test_validate_rejects_a_changed_binding(tmp_path, key, value):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    CK.save(d, epoch=0, global_step=1, model=model, scorer=scorer, opt=opt,
            history=[], cfg={}, binding=binding)
    ck = CK.load(os.path.join(d, CK.LAST))
    bad = CK.validate(ck, dict(binding, **{key: value}))
    assert len(bad) == 1 and bad[0].startswith(key)


def test_validate_can_also_diff_the_config(tmp_path):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    CK.save(d, epoch=0, global_step=1, model=model, scorer=scorer, opt=opt,
            history=[], cfg={"lr": 0.01, "batch_size": 1024}, binding=binding)
    ck = CK.load(os.path.join(d, CK.LAST))
    bad = CK.validate(ck, binding, cfg={"lr": 0.01, "batch_size": 512},
                      strict_cfg=True)
    assert any(m.startswith("cfg.batch_size") for m in bad)


def _run(steps, resume_from=None, ckpt_dir=None, binding=None):
    torch.manual_seed(0)
    model, scorer = Tiny(), Tiny()
    opt = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()),
                           lr=1e-2)
    if resume_from is not None:
        ck = CK.load(resume_from)
        model.load_state_dict(ck["model"])
        scorer.load_state_dict(ck["scorer"])
        opt.load_state_dict(ck["opt"])
        CK.set_rng_state(ck["rng"])
        start = int(ck["epoch"]) + 1
    else:
        start = 0
    for e in range(start, steps):
        _step(model, scorer, opt)
        if ckpt_dir and binding is not None:
            CK.save(ckpt_dir, epoch=e, global_step=e + 1, model=model,
                    scorer=scorer, opt=opt, history=[], cfg={}, binding=binding)
    return model, opt, start


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    _, _, _, binding = _fixture(tmp_path)
    straight, _, _ = _run(4)

    d = str(tmp_path / "ckpt")
    _run(2, ckpt_dir=d, binding=binding)
    resumed, _, start = _run(4, resume_from=os.path.join(d, CK.LAST),
                             ckpt_dir=d, binding=binding)

    assert start == 2, "resume must continue at epoch 2, not restart"
    for a, b in zip(straight.parameters(), resumed.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_dropping_optimizer_state_would_change_the_result(tmp_path):
    _, _, _, binding = _fixture(tmp_path)
    straight, _, _ = _run(4)

    d = str(tmp_path / "ckpt")
    _run(2, ckpt_dir=d, binding=binding)
    ck = CK.load(os.path.join(d, CK.LAST))
    torch.manual_seed(0)
    model, scorer = Tiny(), Tiny()
    model.load_state_dict(ck["model"])
    scorer.load_state_dict(ck["scorer"])
    opt = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()),
                           lr=1e-2)
    _step(model, scorer, opt, n=2)

    assert not all(torch.allclose(a, b, atol=1e-6)
                   for a, b in zip(straight.parameters(), model.parameters()))


def test_global_step_is_continuous_across_a_resume(tmp_path):
    _, _, _, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    _run(2, ckpt_dir=d, binding=binding)
    assert CK.load(os.path.join(d, CK.LAST))["global_step"] == 2
    _run(4, resume_from=os.path.join(d, CK.LAST), ckpt_dir=d, binding=binding)
    ck = CK.load(os.path.join(d, CK.LAST))
    assert ck["epoch"] == 3 and ck["global_step"] == 4


def test_rng_round_trip_restores_all_cpu_streams(tmp_path):
    import random
    s = CK.rng_state()
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    CK.set_rng_state(s)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


def test_resume_state_carries_the_rng(tmp_path):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    CK.save(d, epoch=1, global_step=2, model=model, scorer=scorer, opt=opt,
            history=[], cfg={}, binding=binding)
    rs = CK.to_resume_state(CK.load(os.path.join(d, CK.LAST)))
    assert set(rs) == {"model", "scorer", "opt", "epoch", "rng"}
    assert rs["rng"] is not None and "torch" in rs["rng"]


def test_load_rejects_a_future_checkpoint_version(tmp_path):
    model, scorer, opt, binding = _fixture(tmp_path)
    d = str(tmp_path / "ckpt")
    p = CK.save(d, epoch=0, global_step=1, model=model, scorer=scorer, opt=opt,
                history=[], cfg={}, binding=binding)
    ck = torch.load(p, weights_only=False)
    ck["ckpt_version"] = CK.CKPT_VERSION + 1
    torch.save(ck, p)
    with pytest.raises(CK.IncompatibleCheckpoint):
        CK.load(p)
