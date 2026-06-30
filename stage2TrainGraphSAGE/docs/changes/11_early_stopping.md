# 11 — 验证集早停 + lr/优化器（新增）

**文件：** `stage2TrainGraphSAGE/train.py`（`train` 加 val 早停 + `_val_tau_macro`）+ `ablation.py`（接 flag）
**Phase：** 7

## 问题

`train_diverse.run` 构造了 validation split 但**丢弃**，固定训 25 epoch 并保存**最后一个** epoch，
最佳模型可能早已出现在中间 epoch；且学习率固定 1e-2 未对比更小值。

## 原来代码

```python
# train_diverse.py
train_data, _val, test_data = split_trained_on(data, seed=seed)   # _val 被丢弃
hist, _ = train(model, scorer, ..., epochs=epochs)                # 固定 epoch
tm = eval_perf(model, scorer, test_data, lookup)                  # 保存 final epoch
```

`train()` 原本无 val 评估、无早停、无 best-checkpoint。

## 为什么这么改

guide Phase 7 & 8.1：每 epoch 在固定 val split 上算 `tau_macro`，保存最佳 checkpoint，
用 patience 早停；test 只在最终选择完成后跑一次。并测 lr {3e-4,1e-3,3e-3} 再决定是否保留 1e-2。

## 怎么改

`train()` 加 `val_data/val_lookup/patience/eval_every`：每 epoch（按 eval_every）用
`_val_tau_macro` 在 val 上算 within-dataset 宏 τ，跟踪 best，深拷贝最佳 `state_dict`，
patience 用尽则早停，结束时恢复最佳权重。`ablation.py` 加 `--early_stop/--patience/--lr`，
非 grouped 路径按需注入 val。

## 改后的代码（核心）

```python
def _val_tau_macro(model, scorer, val_data, lookup, device):
    from ...eval_harness import per_dataset_tau
    z = model(val_data.clone().to(device))
    macro, _ = per_dataset_tau(scorer, z, val_data, lookup)
    return macro if macro == macro else -1.0           # NaN -> -1

def train(model, scorer, ..., val_data=None, val_lookup=None, patience=0, eval_every=1):
    best_val, best_state, bad = -2.0, None, 0
    for epoch in range(epochs):
        ... 训练 ...
        if use_val and (epoch % eval_every == 0 or epoch == epochs-1):
            v = _val_tau_macro(model, scorer, val_data, val_lookup, device)
            if v > best_val: best_val, bad, best_state = v, 0, deepcopy((model, scorer).state_dict)
            else:
                bad += 1
                if patience and bad >= patience: break
    if best_state is not None: 恢复 best 权重

# ablation.py
if cfg.get("early_stop"): common.update(val_data=_val, val_lookup=lookup, patience=...)
```

## 实验结论

| 配置 | tau_macro | hit@10 | 备注 |
|---|---|---|---|
| B5_ranknet（25e, final）| 0.398±0.035 | 0.850 | 参考 |
| P7_es（40e, 早停 patience10）| 0.370±0.042 | 0.904 | head↑、稳定 |
| P7_lr3e3（lr 3e-3, 40e）| 0.371±0.022 | 0.849 | 方差最低 |

早停提升了 head retrieval 并更稳；lr 3e-3 降低方差。两者都不大幅改变 tau，作为稳健默认保留。
