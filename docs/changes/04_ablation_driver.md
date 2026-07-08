# 04 — 消融驱动器 ablation.py（新增）

**文件：** 新增 `stage2TrainGraphSAGE/ablation.py`
**Phase：** 0 — 单一实验驱动器（消融阶梯 B0..B9 的承载）。

## 问题

原有训练入口（`experiment.py` / `train_diverse.py`）把一次性配置硬编码进生产默认，
且：同一 seed 同时改 split 和初始化、不产出逐 dataset 指标、不做配对比较、
不报告精确 head retrieval。无法满足"每次只改一处、固定 split 配对比较"的纪律。

## 原来代码

无（新文件）。`experiment.py::run_one` 与 `train_diverse.py::run` 是原一次性驱动器，
保留不动。

## 为什么这么改

guide 要求"建一个专用实验驱动器/manifest，而非把又一个一次性配置塞进生产默认"。
需要一个配置字典驱动、产出 JSON artifact、可被 `compare.py` 配对比较的统一入口，
并把 split_seed 与 init_seed 显式分离。

## 怎么改

新建 `ablation.py`：
- 配置是普通 dict；阶段 flag 随各 Phase 落地逐步加入；未知键回退 B0 基线行为；
- `make_fixed_splits` 每个 split_seed 物化一次、被该 split 下所有 init_seed 复用；
- `init_seed` 仅在模型/scorer 构造与训练前设种，**不改 test 边**；
- 训练后用 `eval_harness` 评测（逐 dataset τ + 精确 head retrieval + collapse + 参与率）；
- 写 `artifacts/ablation/<name>.json`（逐 run、逐 dataset、聚合、head 聚合）；
- `--save_ckpt` 可在首个 run 保存 accepted 候选并做 checkpoint roundtrip（见 #12），
  **绝不覆盖生产 checkpoint**。

## 改后的代码（结构要点）

```python
B0 = dict(num_layers=1, top_frac=0.10, lambda_rank=1.0, lambda_contrast=1.0,
          lambda_mse=0.0, lambda_uniform=0.0, scorer="dot",
          similar_to_mode="dense", similar_to_k=10)        # 后续 Phase flag 陆续加入

def train_eval_one(data, xm0, xd0, cfg, split, *, init_seed, epochs, device):
    train_data, _val, test_data = split
    eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
    ti = cat([train_data[TRAINED_ON].edge_index, eli])     # 训练可见 trained_on（无泄漏）
    M = topk_membership(data, top_frac=cfg["top_frac"], trained_on_index=ti, ...)
    torch.manual_seed(init_seed)                           # init_seed 与 split 分离
    model = HeteroGraphSAGE(..., edge_aware=cfg.get("edge_aware"), separate_heads=..., )
    train_fn = train_grouped if cfg.get("grouped") else train
    hist = train_fn(...)                                   # rank_loss/dm_contrast/early_stop 等
    z_test = model(test_data)                              # 在 test 消息图上算 z
    macro_tau, per_tau = per_dataset_tau(scorer, z_test, test_data, lookup)
    head_macro, per_head = head_retrieval(z_test, test_data, lookup)
    return row, per_tau, per_head, model, scorer

def run(graph_path, cfg, *, name, split_seeds, init_seeds, epochs, device, save_ckpt=None):
    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=...)   # Phase 1 图手术
    for ss in split_seeds:
        split = make_fixed_splits(data, split_seed=ss)
        for isd in init_seeds:
            row, per_tau, per_head, model, _ = train_eval_one(..., init_seed=isd, ...)
            if save_ckpt and 首个run: save_checkpoint + roundtrip 验证
    # 聚合 + head 聚合 -> 写 artifacts/ablation/<name>.json
```

CLI flag（随 Phase 增长）：`--similar_to_mode/--similar_to_k`（P1）、`--edge_aware/--weighted_relations`（P2）、
`--rank_loss/--rank_min_gap`（P4）、`--grouped`（P3）、`--separate_heads`（P5）、
`--lambda_dm_contrast`（P6）、`--early_stop/--patience/--lr`（P7）、`--save_ckpt`。

## 实验结论

所有 B0/B1/B2/B2e_ctrl/B3*/B4/B5/B6/R_*/P6*/P7* 结果均由此驱动器产出，artifact 落在
`artifacts/ablation/`，汇总见 `RESULTS.md`。
