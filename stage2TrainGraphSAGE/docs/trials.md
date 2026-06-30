# Stage 2 Kendall Trials：B0 到 P7_lr3e3

本文汇总 hf1000d/2000m 上用于提升 Kendall tau 和 dataset-to-model 检索质量的
全部主要实验。重点不是重复结果表，而是解释：

- 每个实验从哪个配置出发；
- 它只改变了什么；
- 它想回答什么问题；
- 结果支持或否定了什么结论。

原始汇总见 [changes/RESULTS.md](changes/RESULTS.md)，复现记录见
[changes/RESULTS_REPRODUCE.md](changes/RESULTS_REPRODUCE.md)，逐配置 JSON 位于
`../artifacts/ablation/`。

---

## 1. 共同实验协议

除非单独说明，这些实验使用相同的基础设置：

```text
graph              = hgraph_hf1000d_2000m_xm0_xd0.pt
models             = 2,000
datasets           = 362
split seeds        = 0, 1, 2
initialization     = seed 0
epochs             = 25
lambda_rank        = 1.0
lambda_contrast    = 1.0
scorer             = dot/cosine
GraphSAGE layers   = 1
```

所有配置复用相同的固定 splits，因此可以按相同 test dataset/test edges 做配对
比较。

### 指标方向

| 指标 | 含义 | 越大/越小越好 |
|---|---|---|
| `tau_macro` | 每个 dataset 内模型全排序的 Kendall tau，再对 dataset 等权平均 | 越大越好 |
| `mean_cos` | model embeddings 的平均 pairwise cosine；接近 1 表示更容易 collapse | 通常越小越分散，但不是最终目标 |
| `hit@10` | 真实最佳模型是否进入预测 top-10 | 越大越好 |
| `recall_top3@10` | 真实 top-3 模型中有多少进入预测 top-10 | 越大越好 |
| `ndcg@50` | top-50 的整体排序质量 | 越大越好 |
| `regret@10` | 真实最佳 accuracy 与预测 top-10 中最佳 accuracy 的差距 | 越小越好 |

这些 head metrics 是在每个 test dataset 的 **held-out observed candidates** 中
计算的，不等于从全部 2,000 个模型中做完整 model-lake 检索。

---

## 2. 实验名字如何阅读

```text
B*  = 主 ablation ladder，逐阶段验证 action guide 的假设
R*  = robustness/control/refinement，用来拆解 B5 的收益来源
P6* = Phase 6，直接优化 dataset -> model serving relation
P7* = Phase 7，训练选择与 learning-rate 实验
```

编号不完全代表执行时间。例如 `B5_ranknet` 在表中可能出现在 `B6_heads` 后面；
判断继承关系应看配置，而不是只看编号。

### 实验依赖关系

```text
B0：原始基线
├── B1：删除 similar_to
└── B2：similar_to 改为 top-10、无权重
    └── B2e_ctrl：换 edge-aware/self-path 架构，但不消费边权
        ├── B3_sim：只消费 similar_to 权重
        ├── B3_simtr：再消费 trained_on/reverse 权重
        ├── B3_all：所有 relation 都消费权重
        ├── B4_grouped：改为 full-batch dataset-macro 训练
        ├── B6_heads：model/dataset 改为独立 heads
        └── B5_ranknet：hinge 改为 raw-dot RankNet
            ├── R_tohet：去掉 edge-aware，隔离 RankNet 自身收益
            ├── R_weights：RankNet 下重新打开所有边权
            ├── R_heads：RankNet 下改为独立 heads
            ├── R_mg00：min_gap = 0
            ├── R_mg02：min_gap = 0.02
            ├── P6_dm05/P6_dm10：加入 dataset->model contrastive
            └── P7_es/P7_lr3e3：early stopping / 较低 learning rate

BEST：把当时认为有希望的多项改动一次叠加，不属于干净单变量分支
```

---

## 3. 汇总结果

| name | tau_macro | mean_cos | hit@10 | recall_top3@10 | ndcg@50 | regret@10 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.1928 ± 0.059 | 0.4403 | 0.809 | 0.796 | 0.840 | 0.0055 |
| B1 | 0.2121 ± 0.027 | 0.3400 | 0.785 | 0.777 | 0.840 | 0.0109 |
| B2 | 0.1977 ± 0.064 | 0.3871 | 0.801 | 0.769 | 0.840 | 0.0080 |
| B2e_ctrl | 0.2539 ± 0.054 | 0.3911 | 0.817 | 0.812 | 0.843 | 0.0101 |
| B3_sim | 0.2316 ± 0.060 | 0.3944 | 0.825 | 0.804 | 0.841 | 0.0102 |
| B3_simtr | 0.2664 ± 0.046 | 0.3916 | 0.842 | 0.809 | 0.842 | 0.0084 |
| B3_all | 0.2790 ± 0.022 | 0.3711 | 0.785 | 0.772 | 0.842 | 0.0104 |
| B4_grouped | 0.1746 ± 0.025 | 0.0704 | 0.738 | 0.743 | 0.836 | 0.0200 |
| B6_heads | 0.1722 ± 0.021 | 0.3971 | 0.786 | 0.756 | 0.838 | 0.0099 |
| BEST | 0.1826 ± 0.036 | 0.4192 | 0.738 | 0.757 | 0.834 | 0.0102 |
| **B5_ranknet** | **0.3982 ± 0.035** | 0.3482 | 0.850 | 0.841 | 0.848 | 0.0032 |
| R_tohet | 0.3153 ± 0.031 | 0.3346 | 0.857 | 0.827 | 0.844 | 0.0044 |
| R_weights | 0.3541 ± 0.045 | 0.3381 | 0.857 | 0.844 | 0.846 | 0.0047 |
| R_heads | 0.3826 ± 0.013 | 0.3691 | 0.849 | 0.825 | 0.847 | 0.0035 |
| R_mg00 | 0.3632 ± 0.068 | 0.3427 | 0.889 | 0.873 | 0.848 | 0.0031 |
| **R_mg02** | **0.4029 ± 0.022** | 0.3047 | 0.857 | 0.843 | 0.849 | 0.0031 |
| P6_dm05 | 0.3425 ± 0.065 | 0.2550 | 0.905 | 0.889 | 0.846 | 0.0032 |
| **P6_dm10** | 0.3277 ± 0.070 | 0.2182 | **0.905** | **0.902** | 0.847 | 0.0019 |
| P7_es | 0.3699 ± 0.042 | 0.3465 | 0.904 | 0.862 | 0.846 | 0.0024 |
| P7_lr3e3 | 0.3705 ± 0.022 | 0.3054 | 0.849 | 0.825 | 0.844 | 0.0046 |

---

## 4. Baseline 与 dataset graph 实验

### B0：原始基线

核心配置：

```text
similar_to graph       = Stage 1 构建的 dense graph
GNN                    = PyG to_hetero(SAGEConv)
edge_attr              = 不消费
ranking loss           = hinge on sigmoid(scale * dot + bias)
output head            = model/dataset shared head
training               = shuffled supervision-edge mini-batches
```

研究角色：作为所有后续实验的参考点。

结果：`tau_macro=0.1928`。忽略 ties 时，可粗略理解为约 59.6% 的同数据集模型对
次序判断正确。

结论：embedding 中存在排序信号，但 full-order ranking 较弱。

### B1：删除 `similar_to`

相对 B0 的改动：

```text
similar_to = none
```

研究问题：近乎全连接的 dataset graph 是否正在把不同 datasets 平均到一起？

结果：

```text
tau      0.1928 -> 0.2121
hit@10   0.809  -> 0.785
regret   0.0055 -> 0.0109
```

解释：删除 dense relation 让全排序略有改善，但损害了列表头部。`similar_to` 并非
纯噪声；完全删除不是理想方案。

结论：拒绝作为最终配置，但保留“dense graph 可能轻微有害”的提示。

对应修改说明：[changes/05_graph_surgery.md](changes/05_graph_surgery.md)。

### B2：top-10 unweighted `similar_to`

相对 B0 的改动：

```text
dense similar_to -> 每个 dataset 只保留 top-10 邻居
保留的边全部等权
```

研究问题：使用少量强邻居，是否比近乎全连接更好？

结果：`tau_macro=0.1977`，与 B0 基本相当；head metrics 略差。

结论：单纯 pruning 不是主要性能杠杆。它提供了更合理、可控的图结构，但不能单独
解决 Kendall 问题。

---

## 5. Edge-aware 架构与 edge weights 实验

### B2e_ctrl：edge-aware architecture control

相对 B2 的改动：

```text
edge_aware=True
weighted_relations=[]
```

这里故意不使用任何 numerical edge weight。变化仅来自新卷积结构：

- 显式 self/residual path；
- 每个 relation 独立卷积；
- relation-specific gates；
- 新的邻居归一化实现。

研究问题：后续收益究竟来自 edge-aware 架构，还是来自 edge weights？

结果：`tau 0.1977 -> 0.2539`，hit@10 也略升。

结论：新架构本身有明确收益；之后必须以它为 control，不能把所有提升归因给边权。

对应实现：[changes/02_edge_aware_message_passing.md](changes/02_edge_aware_message_passing.md)。

### B3_sim：只消费 `similar_to` weight

相对 B2e_ctrl 的改动：

```text
weighted_relations = [similar_to]
```

研究问题：Stage 1 保存的 dataset similarity 数值能否改善消息聚合？

结果：`tau 0.2539 -> 0.2316`；hit@10 略升，但总体没有净收益。

结论：similarity weight 本身没有改善 Kendall。保留 edge-aware 架构，但不接受这项
权重配置。

### B3_simtr：再加入 performance weights

相对 B3_sim 的改动：

```text
weighted_relations = [similar_to, trained_on, rev_trained_on]
```

研究问题：训练可见 performance accuracy 作为消息权重是否有帮助？

结果：`tau=0.2664`、`hit@10=0.842`。在 hinge regime 下，它比只加
`similar_to` weight 更好。

结论：performance weights 在旧 hinge loss 下有一些信号，但仍需在最终 RankNet
regime 中重新验证，不能直接接受。

### B3_all：所有 relations 都消费权重

配置：

```text
weighted_relations = None
```

在当前实现中，`None` 表示所有带 `edge_attr` 的 relations 都使用权重，包括
`similar_to`、performance 和 lineage。

研究问题：所有可用的关系强度是否应统一进入消息传播？

结果：`tau=0.2790`，是 hinge + edge-aware 系列中的最高值；但
`hit@10=0.785`、`regret@10=0.0104`，head 明显变差。

结论：提高 full-order tau 不代表候选生成更好。该配置被拒绝。

---

## 6. Batch reduction 与 projection head 实验

### B4_grouped：full-batch dataset-macro training

相对 B2e_ctrl 的改动：

```text
grouped=True
ranking loss 先在每个 dataset 内平均，再对 datasets 平均
每个 epoch 使用一次 full-graph optimization step
```

研究问题：既然 `tau_macro` 对 dataset 等权，训练 loss 是否也应严格对 dataset
等权？

结果：

```text
tau        = 0.1746
mean_cos   = 0.0704
hit@10     = 0.738
regret@10  = 0.0200
```

embedding 看起来非常分散，但实际排序全面变差。

结论：当前 full-batch grouped 实现被拒绝。不过它同时改变了 batch 随机性、每个
epoch 的 optimizer-step 数和 edge-dropout 行为，因此不能据此证明“macro-balanced
sampling”这个思想本身错误。

对应说明：[changes/08_grouped_training.md](changes/08_grouped_training.md)。

### B6_heads：separate model/dataset heads under hinge

相对 B2e_ctrl 的改动：

```text
shared output head -> model_head + dataset_head
```

研究问题：model 和 dataset 上游语义不同，独立 projection 是否更容易对齐？

结果：`tau=0.1722`，head metrics 同时下降。

结论：在 hinge regime 下，separate heads 没有帮助。后续仍需在 RankNet regime
重新做 control。

对应说明：[changes/09_separate_heads.md](changes/09_separate_heads.md)。

### BEST：预先堆叠的组合实验

配置同时包含：

```text
top-10 graph
edge-aware
all edge weights
RankNet (min_gap=0.01)
separate heads
grouped full-batch training
```

研究问题：把当时认为可能有益的修改全部叠加，能否得到最佳配置？

结果：`tau=0.1826`，hit@10 仅 0.738。

结论：失败。`BEST` 是运行前的候选名称，不是结果意义上的 best。这个实验说明多个
组件存在强 interaction，不能把单项好点子直接全部叠加，也证明了逐项 ablation 的
必要性。

---

## 7. Ranking-loss 实验：主要突破

### B5_ranknet：raw-dot RankNet

相对 B2e_ctrl 的核心改动：

```text
hinge(sigmoid(scale * dot + bias))
    -> raw-dot RankNet

rank_temperature  = 0.1
rank_min_gap      = 0.01
edge weights      = OFF
shared head       = ON
mini-batch        = ON
```

RankNet 直接优化 HNSW 将使用的 normalized dot/cosine geometry：

\[
\mathcal L_d=\operatorname{mean}_{m^+\succ m^-}
\operatorname{softplus}
\left(-\frac{z_{m^+}^{\top}z_d-z_{m^-}^{\top}z_d}{T}\right).
\]

`min_gap=0.01` 表示真实 accuracy 差距不超过 0.01 的模型对不作为严格排序目标。

研究问题：B0 的主要瓶颈是否其实是与 serving geometry 不完全一致的 hinge
surrogate？

结果：

```text
B2e_ctrl tau   0.2539 -> 0.3982
hit@10                  0.817 -> 0.850
regret@10               0.0101 -> 0.0032
```

结论：接受。Ranking loss 是本轮实验中最大的单一性能杠杆。

实现说明：[changes/07_ranknet_loss.md](changes/07_ranknet_loss.md)。

---

## 8. RankNet 收益拆解与 refinement

### R_tohet：RankNet without edge-aware

相对 B5 的改动：

```text
edge_aware=True -> False
回到原始 to_hetero(SAGEConv)
RankNet 和 top-10 graph 保持不变
```

研究问题：B5 的提升中，有多少来自 RankNet 自身，有多少来自 edge-aware
self/residual 架构？

结果：`tau=0.3153`。

可做两个配对理解：

```text
B2 -> R_tohet：0.1977 -> 0.3153，约 +0.12 来自 RankNet
R_tohet -> B5：0.3153 -> 0.3982，约 +0.08 来自 edge-aware 架构
```

结论：RankNet 是最大贡献者，edge-aware self-path 是次级但明显的贡献者。

### R_weights：RankNet + all edge weights

相对 B5 的改动：

```text
weighted_relations = None
```

研究问题：在 hinge 下略有帮助的 edge weights，在 RankNet 下是否仍有帮助？

结果：`tau 0.3982 -> 0.3541`，head 也没有形成优势。

结论：边权的收益依赖 loss regime；在 winning RankNet regime 中应关闭 edge
weights。保留 edge-aware 架构并不等于必须消费 edge values。

### R_heads：RankNet + separate heads

相对 B5 的改动：

```text
separate_heads=False -> True
```

研究问题：B6 的 separate-head 失败是否只是 hinge loss 造成的？

结果：`tau=0.3826±0.013`，均值略低于 B5，但方差较小；head 基本持平。

结论：独立 heads 在 RankNet 下是 neutral/slight regression，没有足够证据替换
shared head。

### R_mg00：不设置 accuracy tie threshold

相对 B5 的改动：

```text
rank_min_gap = 0.00
```

研究问题：是否应该对所有非完全相等的 accuracy 强制排序？

结果：`tau=0.3632±0.068`，方差最大；`hit@10=0.889` 较高。

解释：它更积极地优化列表头部，但会把微小、可能属于测量噪声的 accuracy 差异也
当作严格 preference，导致 full-order tau 不稳定。

结论：不接受为 ranking-priority 配置。

### R_mg02：更严格的 tie threshold

相对 B5 的改动：

```text
rank_min_gap = 0.02
```

研究问题：把 accuracy 差距不超过 2 个百分点的模型视为近似同分，能否减少噪声？

结果：

```text
tau_macro = 0.4029 ± 0.022
hit@10    = 0.857
regret@10 = 0.0031
```

这是最高 tau 且方差最低的 RankNet 配置。

结论：接受为当前 ranking-priority winner。对应保存 checkpoint 的单次运行叫
`ACCEPTED_ranknet`；它不是另一个新算法。

---

## 9. Phase 6：直接优化 serving relation

### P6_dm05：加入较弱 dataset-to-model contrastive

相对 B5 的改动：

```text
lambda_dm_contrast = 0.5
```

该辅助目标直接将 dataset embedding 拉向 training-visible top-performing models，
并远离已知低性能 models。

研究问题：直接训练 `z_d -> z_m`，是否能提高 candidate-generation head recall？

结果：

```text
tau_macro        = 0.3425
hit@10           = 0.905
recall_top3@10   = 0.889
```

结论：top-10 明显变强，但完整排序变差。

### P6_dm10：更强 dataset-to-model contrastive

相对 B5 的改动：

```text
lambda_dm_contrast = 1.0
```

结果：

```text
tau_macro        = 0.3277
hit@10           = 0.905
recall_top3@10   = 0.902
regret@10        = 0.0019
```

结论：这是最明显的 serving/head-priority 配置，但代价是 full-order Kendall
显著下降。它与 R_mg02 形成 Pareto tradeoff：

```text
R_mg02   = 更好的完整排序
P6_dm10  = 更好的 top-10 candidate recall
```

若 HNSW 只输出很小的 K，它可能更有价值；若后续 reranker 接收 K=50，R_mg02 的
完整排序表现通常更稳妥。

实现说明：[changes/10_dataset_to_model_contrastive.md](changes/10_dataset_to_model_contrastive.md)。

---

## 10. Phase 7：训练选择与 learning rate

### P7_es：validation early stopping

相对 B5 的训练改动：

```text
max epochs      = 40
learning rate   = 0.01
early stopping  = ON
selection       = validation tau_macro
patience        = 10
```

研究问题：固定训练 25 epochs 是否错过了更好的 validation checkpoint？

结果：

```text
tau_macro = 0.3699
hit@10    = 0.904
```

结论：early stopping 产生了更强的 top-10，但没有超过 B5/R_mg02 的 full-order
tau。它证明 validation selection 可用，但当前 selection metric/训练曲线没有带来
ranking-priority winner。

### P7_lr3e3：较低 learning rate、较长训练

相对 B5 的训练改动：

```text
learning rate   = 0.003
epochs          = 40
early stopping  = OFF
```

研究问题：原来的 `lr=0.01, epochs=25` 是否过于激进？

结果：`tau=0.3705±0.022`，方差较低，但均值和 head metrics 都没有超过 B5。

结论：降低 learning rate 没有形成净收益，不接受为当前默认。

实现说明：[changes/11_early_stopping.md](changes/11_early_stopping.md)。

---

## 11. 从全部 trials 得出的结论

### 已得到较强证据的结论

1. **主要瓶颈是 ranking surrogate。** Raw-dot RankNet 带来的收益远大于图剪枝、
   edge weights、heads 或常规训练调整。
2. **edge-aware 架构有效，但主要不是因为边权。** Self/residual relation-aware
   structure 带来次级收益；数值 edge weights 在最终 RankNet regime 中反而有害。
3. **`min_gap=0.02` 有效过滤了近似同分噪声。** 它给出最高 tau 和最低 split
   variance。
4. **优化 full-order ranking 和优化 top-K serving 不是同一件事。** Dataset-to-model
   contrastive 提高 top-10，但显著牺牲 Kendall。
5. **低 mean cosine 不保证模型更好。** B4 的 `mean_cos=0.0704` 看似非常分散，
   但其排序和检索最差之一。

### 被当前实验否定或未支持的假设

- 单纯删除或 top-k pruning dataset graph 不能显著解决排序问题；
- 所有 edge weights 都进入消息传播不是更好；
- separate model/dataset heads 没有超过 shared head；
- 当前 full-batch grouped 实现没有帮助；
- early stopping 和较低 learning rate 没有超过原始 RankNet schedule。

### 当前两个有意义的 operating points

| 使用目标 | 推荐配置 | 核心指标 |
|---|---|---|
| ranking-priority | R_mg02：edge-aware unweighted + RankNet + `min_gap=0.02` | `tau=0.4029±0.022` |
| serving/head-priority | P6_dm10：B5 + `lambda_dm_contrast=1.0` | `hit@10=0.905`, `recall_top3@10=0.902` |

---

## 12. 阅读这些结果时的限制

1. 三次运行改变的是 split seed，只有一个 initialization seed；当前标准差不包含完整
   initialization variance。
2. Head retrieval 的候选集合是 held-out observed models，不是全部 2,000 个模型；
   不能把 `hit@10=0.905` 直接称为 full-lake top-10 recall。
3. 当前是 transductive edge split；真正新 dataset 没有 `trained_on` message edges，
   还需要 cold-dataset evaluation。
4. B4 同时改变了 macro reduction、batching、optimizer-step 数和 dropout 行为，所以
   它否定的是当前 full-batch 实现，不是所有 dataset-balanced sampling 方法。

因此，这批 trials 最可靠的结论是：**在当前 transductive、observed-candidate 评估
设置下，raw-dot RankNet 是决定性的提升来源；R_mg02 是 ranking winner，P6_dm10
是一个明确但有代价的 serving Pareto variant。**
