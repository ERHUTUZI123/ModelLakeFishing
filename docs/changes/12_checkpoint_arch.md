# 12 — checkpoint 架构持久化（兼容 edge_aware / separate_heads）

**文件：** `stage2TrainGraphSAGE/learnable.py`（`_arch_of` / `load_checkpoint`）
**Phase：** 2 / 5 配套（guide 必需测试 #10：checkpoint roundtrip 保留两个投影头并复现 embedding）

## 问题

`_arch_of` 通过 `model.head.out_features` 读 `out_dim`，并未记录新加的 `edge_aware` /
`weighted_relations` / `separate_heads`。若直接给 edge-aware 或 separate-heads 模型存档，
重建时会用错架构、`load_state_dict` 键不匹配或 `model.head` 不存在 → roundtrip 失败。

## 原来代码

```python
def _arch_of(model):
    arch = {
        ...
        "hidden_channels": model.model_proj.out_features,
        "out_dim": model.head.out_features,              # separate_heads 时 model.head 不存在
        "num_layers": getattr(model, "num_layers", 2),
        # 无 edge_aware / weighted_relations / separate_heads
    }

def load_checkpoint(path, ...):
    model = HeteroGraphSAGE(..., num_layers=arch.get("num_layers", 2),
                            # 无 edge_aware / separate_heads 重建参数
                            )
```

## 为什么这么改

新增的架构开关必须随 checkpoint 持久化，否则 accepted 候选无法被无损重载，
也无法通过 guide 要求的 checkpoint roundtrip 测试。需向后兼容旧 checkpoint（默认 False）。

## 怎么改

`_arch_of` 记录 `edge_aware`、`weighted_relations`、`separate_heads`，并在 separate_heads 时
从 `model_head` 读 `out_dim`；`load_checkpoint` 把这些传回构造函数（`arch.get(..., 默认)`
保旧档有效）。

## 改后的代码（核心）

```python
def _arch_of(model):
    arch = {
        ...
        "hidden_channels": model.model_proj.out_features,
        "out_dim": (model.model_head.out_features
                    if getattr(model, "separate_heads", False) else model.head.out_features),
        "separate_heads": getattr(model, "separate_heads", False),
        "num_layers": getattr(model, "num_layers", 2),
        "edge_aware": getattr(model, "edge_aware", False),
        "weighted_relations": getattr(model, "weighted_relations", None),
    }
    ...

def load_checkpoint(path, *, map_location="cpu"):
    model = HeteroGraphSAGE(
        ..., num_layers=arch.get("num_layers", 2),
        edge_aware=arch.get("edge_aware", False),
        weighted_relations=arch.get("weighted_relations", None),
        separate_heads=arch.get("separate_heads", False),
        num_task_types=arch.get("num_task_types"), ...)
    model.load_state_dict(ckpt["state_dict"]); ...
```

## 实验结论

accepted 候选 `stage2_hf1000d_ranknet_candidate.pt`（edge_aware=True）存档 + roundtrip 验证 OK
（`[ckpt] saved ... -> OK (roundtrip matches)`），并能被 `diverse_diagnostics` 正确重载
（num_layers=1、edge_aware 生效）。对应测试：`tests/test_phase2_edge_aware.py` 的 roundtrip 用例。
